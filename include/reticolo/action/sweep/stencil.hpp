#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/parallel.hpp>
#include <reticolo/core/site.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>

// Function-scope FP-reassociation hint for reduction loops. On clang this
// breaks the strict left-to-right `total += body(i)` dependency chain so
// independent body evaluations (sin/cos/heavy fma) can overlap. Other
// compilers still build; they just lose this particular micro-opt.
#if defined(__clang__)
    #define RETICOLO_FP_REASSOCIATE _Pragma("clang fp reassociate(on)")
#else
    #define RETICOLO_FP_REASSOCIATE
#endif

// Nearest-neighbour traversal engine, shared by every NN scalar family (site,
// bond, complex). It sweeps the lattice as a stack of innermost-axis rows: dim 0
// (x) is kept FULL and contiguous so the inner loop vectorises on every
// architecture (NEON / SSE / AVX2 / AVX-512 / SVE) with no intrinsics; the outer
// axes (dims 1..D-1) are an explicit hoisted loop nest, one per dimension
// (`stencil_map_{1,2,3,4}d_`, `stencil_reduce_*`). Each row hoists its neighbour
// bases as scalars and sums the per-site neighbours INLINE in the dim-0 loop, so
// the compiler collapses the whole kernel.
//
// The two entry points:
//
//   visit_stencil<Policy>(l, comb, body):    map. body(i, self, agg) -> void; the
//                                    body owns the write (out[i]=… or m[i]+=…).
//   reduce_stencil<Policy>(l, comb, body):   reduce. body(self, agg) -> Acc,
//                                    accumulated into a total that is deterministic
//                                    for a fixed (team, slabs) config.
//
// Which neighbours fold into `agg` is a compile-time Policy (AllDirs / FwdOnly /
// BwdOnly, below). Force uses AllDirs (all 2·ndims neighbours); s_full uses
// FwdOnly (the ndims forward neighbours, each bond once).
//
// `comb(self, nbr_value)` is applied per neighbour and the results summed into
// `agg` — the one variation point across families:
//   * site  → IdentityCombine (agg = Σ neighbour values; the leaf's kernel then
//             consumes self + the summed neighbours);
//   * bond  → the leaf's bond kernel (agg = Σ f(self, nbr), an endpoint-difference
//             energy that cannot be pre-summed).
//
// Work items come from the field's canonical partition (exec::partition):
// contiguous outer-dim slabs shared by EVERY per-site op, so thread↔memory
// ownership is identical across all passes of a trajectory. Every item is a
// contiguous span, keeping the hardware prefetch streams intact — measured
// (Apple M and 32-core Linux x86, every thread count) this beats 512 KB cache
// tiling, which was dropped. The map output is write-disjoint (bit-identical to
// the gather fallback, any thread or slab count); the reduce folds the partition
// in canonical item order — deterministic for a fixed (team, slabs-per-thread),
// but the partition now tracks the team size, so a different team re-folds.
//
// D in {1, 2, 3, 4} take the vectorised generic path; D > 4 falls back to a flat
// gather through the neighbour table (exact, just slower). Raising the vectorised
// ceiling is a one-line switch case — the stencil body already handles any D.
//
// Compile with -DRETICOLO_HOT_LOOP_FORCE_FALLBACK=1 to force every call onto the
// gather fallback regardless of ndims (the "old hot loop" bench path).

namespace reticolo::action::sweep {

// ---------- neighbour policy + combine ---------------------------------------

// Which nearest neighbours fold into a site's `agg`, as compile-time tags:
//   AllDirs → both ±mu (2·ndims neighbours)  — force / full stencil
//   FwdOnly → +mu only (ndims, each bond once) — s_full
//   BwdOnly → −mu only  — the mirror of FwdOnly
// Selected at instantiation, so the `if constexpr` branches below cost nothing.
struct AllDirs {
    static constexpr bool fwd = true;
    static constexpr bool bwd = true;
};
struct FwdOnly {
    static constexpr bool fwd = true;
    static constexpr bool bwd = false;
};
struct BwdOnly {
    static constexpr bool fwd = false;
    static constexpr bool bwd = true;
};

// The site combine: fold raw neighbour values (self ignored). Inlines away so the
// site path is exactly `agg += data[nbr]` — bit-identical to the pre-generic loop.
struct IdentityCombine {
    template <class T>
    [[gnu::always_inline]] T operator()(T /*self*/, T nbr) const noexcept {
        return nbr;
    }
};

// Left-associative fold of comb(self, data[i])… over the neighbour indices `i`,
// SEEDED FROM THE FIRST TERM (no `T{0}` seed). A unary left fold `(... + expr)`
// expands to ((c₀ + c₁) + c₂) + …, i.e. exactly the hand-written a+b+c+… the
// vectoriser widens across the dim-0 loop with no reassociation (works without
// fast-math). Seeding from `T{0}` instead would add a `0 + c₀` that clang cannot
// fold away (signed zero) — one extra fadd per site on the critical chain.
// Never called with an empty pack (every site has ≥1 neighbour).
template <class T, class Comb, class... I>
[[gnu::always_inline]] inline T nbr_sum(Comb const& comb, T const* data, T self, I... i) noexcept {
    return (... + comb(self, data[i]));
}

// Sweep the dim-0 columns x∈[0,L0) of one row, invoking at(x, xm, xp) with the
// wrapped dim-0 neighbour columns. Only the ends the Policy actually reads are
// peeled — AllDirs both, FwdOnly the +x end, BwdOnly the −x end — leaving the
// middle branch-free with LITERAL loop bounds. Emitting the bounds as literals
// (not a runtime `for(x=xs; x<xh)`) is load-bearing: the constexpr-folded runtime
// form costs the scalar peel extra address arithmetic + spills, ~9% on the force.
template <class Policy, class At>
[[gnu::always_inline]] inline void x_sweep_(std::size_t L0, At const& at) noexcept {
    if constexpr (Policy::fwd && Policy::bwd) {
        at(0, L0 - 1, 1);
        for (std::size_t x = 1; x + 1 < L0; ++x) {
            at(x, x - 1, x + 1);
        }
        at(L0 - 1, L0 - 2, 0);
    } else if constexpr (Policy::fwd) {
        for (std::size_t x = 0; x + 1 < L0; ++x) {
            at(x, x, x + 1);
        }
        at(L0 - 1, L0 - 1, 0);
    } else {
        at(0, L0 - 1, 0);
        for (std::size_t x = 1; x < L0; ++x) {
            at(x, x - 1, x);
        }
    }
}

// ---------- gather fallback (any dimension) ----------------------------------

// Flat neighbour-table sweep over sites [s0, s0+cnt). Per axis mu (ascending) it
// reads next[mu] iff Policy::fwd and prev[mu] iff Policy::bwd, in that order — the
// canonical order the vectorised nests reproduce. (This runtime-length loop must
// accumulate from T{0}; the nests fold from the first term. For real field data
// the two agree bit-for-bit — signed-zero neighbour values never arise.)
template <class Policy, class T, class Comb, class Body>
inline void stencil_map_flat_(Lattice<T> const& l,
                              std::size_t s0,
                              std::size_t cnt,
                              Comb const& comb,
                              Body const& body) noexcept {
    auto const& idx                               = l.indexing_ref();
    T const* data                                 = l.data();
    [[maybe_unused]] Site::value_type const* next = idx.next_data();
    [[maybe_unused]] Site::value_type const* prev = idx.prev_data();
    std::size_t const d                           = idx.ndims();
    std::size_t const end                         = s0 + cnt;

    for (std::size_t i = s0; i < end; ++i) {
        T const self           = data[i];
        T agg                  = T{0};
        std::size_t const base = i * d;
        for (std::size_t mu = 0; mu < d; ++mu) {
            if constexpr (Policy::fwd) {
                agg += comb(self, data[next[base + mu]]);
            }
            if constexpr (Policy::bwd) {
                agg += comb(self, data[prev[base + mu]]);
            }
        }
        body(i, self, agg);
    }
}

template <class Policy, class Acc, class T, class Comb, class Body>
[[nodiscard]] inline Acc stencil_reduce_flat_(Lattice<T> const& l,
                                              std::size_t s0,
                                              std::size_t cnt,
                                              Comb const& comb,
                                              Body const& body) noexcept {
    RETICOLO_FP_REASSOCIATE
    auto const& idx                               = l.indexing_ref();
    T const* data                                 = l.data();
    [[maybe_unused]] Site::value_type const* next = idx.next_data();
    [[maybe_unused]] Site::value_type const* prev = idx.prev_data();
    std::size_t const d                           = idx.ndims();
    std::size_t const end                         = s0 + cnt;

    Acc total{};
    for (std::size_t i = s0; i < end; ++i) {
        T const self           = data[i];
        T agg                  = T{0};
        std::size_t const base = i * d;
        for (std::size_t mu = 0; mu < d; ++mu) {
            if constexpr (Policy::fwd) {
                agg += comb(self, data[next[base + mu]]);
            }
            if constexpr (Policy::bwd) {
                agg += comb(self, data[prev[base + mu]]);
            }
        }
        total += body(self, agg);
    }
    return total;
}

// Identity-combine fallbacks, kept under their historical names for the site unit
// tests (which use them as the bit-exact reference) and the site D>4 path: force
// reads all neighbours, s_full the forward ones.
template <class T, class Body>
inline void visit_nn_fallback_(Lattice<T> const& l,
                               std::size_t s0,
                               std::size_t cnt,
                               Body const& body) noexcept {
    stencil_map_flat_<AllDirs, T>(l, s0, cnt, IdentityCombine{}, body);
}

template <class T, class Acc = T, class Body>
[[nodiscard]] inline Acc reduce_fwd_fallback_(Lattice<T> const& l,
                                              std::size_t s0,
                                              std::size_t cnt,
                                              Body const& body) noexcept {
    return stencil_reduce_flat_<FwdOnly, Acc, T>(l, s0, cnt, IdentityCombine{}, body);
}

// Build (L, stride) geometry arrays for a D-dim lattice. stride[0]=1,
// stride[mu]=∏_{k<mu} L[k].
template <std::size_t D, class T>
[[nodiscard]] inline std::pair<std::array<std::size_t, D>, std::array<std::size_t, D>>
geometry_(Lattice<T> const& l) noexcept {
    auto const& sh = l.shape();
    std::array<std::size_t, D> L{};
    std::array<std::size_t, D> stride{};
    for (std::size_t mu = 0; mu < D; ++mu) {
        L[mu] = sh[mu];
    }
    stride[0] = 1;
    for (std::size_t mu = 1; mu < D; ++mu) {
        stride[mu] = stride[mu - 1] * L[mu - 1];
    }
    return {L, stride};
}

// ---------- dispatch ---------------------------------------------------------

// Map (Acc = void) or reduce (Acc = fold type) selector, so one dispatcher serves
// both. `run_items_` splits an abstract work-item count; `run_ranges_` splits a
// flat [0,n) into gran-aligned chunks.
template <class Acc, class Work>
inline Acc run_items_(int nthreads, std::size_t n_items, Work const& work) {
    if constexpr (std::is_void_v<Acc>) {
        reticolo::exec::parallel_map(nthreads, n_items, work);
    } else {
        return reticolo::exec::parallel_reduce<Acc>(nthreads, n_items, work);
    }
}

template <class Acc, class Range>
inline Acc run_ranges_(std::size_t n, std::size_t bps, std::size_t gran, Range const& r) {
    if constexpr (std::is_void_v<Acc>) {
        reticolo::exec::parallel_map_ranges(n, bps, gran, r);
    } else {
        return reticolo::exec::parallel_reduce_ranges<Acc>(n, bps, gran, r);
    }
}

// Run every item of the field's canonical partition (exec::partition) through the
// D-templated item driver. Item index → the [lo, hi) sub-lattice bounds of the
// dims the partition splits (dims [split_dim, D), innermost of them fastest —
// matching the flat span order, so a reduce folds partials in memory order).
// Using the SAME partition as every elementwise op is the alignment guarantee:
// thread k's static block is the same contiguous slab in every pass.
template <class Acc, std::size_t D, class T, class Item>
inline Acc run_partition_items_(Lattice<T> const& l, int nthreads, Item const& item) {
    auto const [L, stride]  = geometry_<D>(l);
    exec::Partition const p = reticolo::exec::partition(l);
    reticolo::exec::note_slicing(L.data(), D, l.bytes_per_site(), p, nthreads);
    return run_items_<Acc>(nthreads, p.n_items, [&](std::size_t it) {
        std::array<std::size_t, D> lo{};
        std::array<std::size_t, D> hi{};
        for (std::size_t mu = 0; mu < D; ++mu) {
            hi[mu] = L[mu];
        }
        std::size_t rem = it;
        // split_dim is cut into blocks of `p.block` (block-index enumerated
        // fastest → contiguous flat spans); dims above it are single-indexed.
        std::size_t const nblk = L[p.split_dim] / p.block;
        std::size_t const cb   = rem % nblk;
        rem /= nblk;
        lo[p.split_dim] = cb * p.block;
        hi[p.split_dim] = (cb + 1) * p.block;
        for (std::size_t mu = p.split_dim + 1; mu < D; ++mu) {
            std::size_t const c = rem % L[mu];
            rem /= L[mu];
            lo[mu] = c;
            hi[mu] = c + 1;
        }
        return item.template operator()<D>(L, stride, lo, hi);
    });
}

// The single ndims → work-item dispatch, shared by the site stencil AND the complex
// split-last drivers, for both map (Acc = void) and reduce. It routes D∈{2,3,4}
// through the field's canonical partition (exec::partition — the same items every
// elementwise op uses), 1D through inner chunks, and D>4 through the flat gather
// fallback; the caller supplies only the per-item behaviour:
//   item.template operator()<D>(L, stride, lo, hi) — one partition item, D∈{2,3,4}
//   one_d(x0, cnt)  — one inner-axis chunk of a 1D lattice
//   flat(s0, cnt)   — a flat neighbour-table gather (D>4 / forced fallback)
// each returning void (map) or Acc (reduce). The tile grid, item→(lo,hi) decode and
// the row/plane ranges live here once instead of in four near-identical switches.
template <class Acc, class T, class Item, class OneD, class Flat>
inline Acc traverse_dispatch_(Lattice<T> const& l,
                              [[maybe_unused]] Item const& item,
                              [[maybe_unused]] OneD const& one_d,
                              Flat const& flat) {
    std::size_t const n                 = l.nsites();
    std::size_t const bps               = l.bytes_per_site();
    [[maybe_unused]] int const nthreads = reticolo::exec::traverse_threads(n, bps);
#if RETICOLO_HOT_LOOP_FORCE_FALLBACK
    return run_ranges_<Acc>(n, bps, 1, flat);
#else
    switch (l.ndims()) {
        case 1:
            return run_ranges_<Acc>(l.shape()[0], bps, 1, one_d);
        case 2:
            return run_partition_items_<Acc, 2>(l, nthreads, item);
        case 3:
            return run_partition_items_<Acc, 3>(l, nthreads, item);
        case 4:
            return run_partition_items_<Acc, 4>(l, nthreads, item);
        default:
            return run_ranges_<Acc>(n, bps, 1, flat);
    }
#endif
}

// ---------- hand-written per-dimension item nests ----------------------------
//
// One explicit hoisted loop nest per D over a partition item [lo,hi) (inner dim 0
// kept full). Each outer level hoists its neighbour row/plane/hypercube bases as
// scalars; the innermost dim-0 loop folds the Policy-selected neighbours INLINE
// (via `nbr_sum`: self at data[i], the ±x contiguous, the outer bases at
// `base + x`) and calls `body`. Neighbour order is per-axis fwd-before-bwd
// (x⁺,x⁻,y⁺,y⁻,…) — the exact gather-fallback fold order → bit-identical for any
// thread count. The x-wrap ends are peeled: the +x end iff Policy::fwd, the −x end
// iff Policy::bwd, leaving the middle dim-0 loop branch-free. Wraps use the full
// L[mu].

// 1D: a chunk [x0, xe) of the single axis. Only the touched global x-wraps are
// peeled (−x end iff bwd, +x end iff fwd); a chunk clear of them runs branch-free.
template <class Policy, class T, class Comb, class Body>
inline void stencil_map_1d_(T const* data,
                            std::size_t x0,
                            std::size_t xe,
                            std::size_t L0,
                            Comb const& comb,
                            Body const& body) noexcept {
    auto const at = [&](std::size_t i, std::size_t xm, std::size_t xp) {
        T const self = data[i];
        T const agg  = [&] {
            if constexpr (Policy::fwd && Policy::bwd) {
                return nbr_sum(comb, data, self, xp, xm);
            } else if constexpr (Policy::fwd) {
                return nbr_sum(comb, data, self, xp);
            } else {
                return nbr_sum(comb, data, self, xm);
            }
        }();
        body(i, self, agg);
    };
    std::size_t xb   = x0;
    std::size_t xend = xe;
    if constexpr (Policy::bwd) {
        if (x0 == 0) {
            at(0, L0 - 1, 1);
            xb = 1;
        }
    }
    if constexpr (Policy::fwd) {
        if (xe == L0) {
            xend = L0 - 1;
        }
    }
    for (std::size_t x = xb; x < xend; ++x) {
        at(x, x - 1, x + 1);
    }
    if constexpr (Policy::fwd) {
        if (xe == L0) {
            at(L0 - 1, L0 - 2, 0);
        }
    }
}

template <class Policy, class Acc, class T, class Comb, class Body>
[[nodiscard]] inline Acc stencil_reduce_1d_(T const* data,
                                            std::size_t x0,
                                            std::size_t xe,
                                            std::size_t L0,
                                            Comb const& comb,
                                            Body const& body) noexcept {
    Acc total{};
    auto const at = [&](std::size_t i, std::size_t xm, std::size_t xp) {
        T const self = data[i];
        T const agg  = [&] {
            if constexpr (Policy::fwd && Policy::bwd) {
                return nbr_sum(comb, data, self, xp, xm);
            } else if constexpr (Policy::fwd) {
                return nbr_sum(comb, data, self, xp);
            } else {
                return nbr_sum(comb, data, self, xm);
            }
        }();
        total += body(self, agg);
    };
    std::size_t xb   = x0;
    std::size_t xend = xe;
    if constexpr (Policy::bwd) {
        if (x0 == 0) {
            at(0, L0 - 1, 1);
            xb = 1;
        }
    }
    if constexpr (Policy::fwd) {
        if (xe == L0) {
            xend = L0 - 1;
        }
    }
    for (std::size_t x = xb; x < xend; ++x) {
        at(x, x - 1, x + 1);
    }
    if constexpr (Policy::fwd) {
        if (xe == L0) {
            at(L0 - 1, L0 - 2, 0);
        }
    }
    return total;
}

template <class Policy, class T, class Comb, class Body>
inline void stencil_map_2d_(T const* data,
                            std::array<std::size_t, 2> const& L,
                            std::array<std::size_t, 2> const& stride,
                            std::array<std::size_t, 2> const& lo,
                            std::array<std::size_t, 2> const& hi,
                            Comb const& comb,
                            Body const& body) noexcept {
    std::size_t const L0 = L[0];
    std::size_t const L1 = L[1];
    std::size_t const s1 = stride[1];
    for (std::size_t y = lo[1]; y < hi[1]; ++y) {
        std::size_t const yp     = (y + 1 == L1) ? 0 : (y + 1);
        std::size_t const ym     = (y == 0) ? (L1 - 1) : (y - 1);
        std::size_t const row    = y * s1;
        std::size_t const row_yp = yp * s1;
        std::size_t const row_ym = ym * s1;
        x_sweep_<Policy>(L0, [&](std::size_t x, std::size_t xm, std::size_t xp) {
            std::size_t const i = row + x;
            T const self        = data[i];
            if constexpr (Policy::fwd && Policy::bwd) {
                body(
                    i, self, nbr_sum(comb, data, self, row + xp, row + xm, row_yp + x, row_ym + x));
            } else if constexpr (Policy::fwd) {
                body(i, self, nbr_sum(comb, data, self, row + xp, row_yp + x));
            } else {
                body(i, self, nbr_sum(comb, data, self, row + xm, row_ym + x));
            }
        });
    }
}

template <class Policy, class Acc, class T, class Comb, class Body>
[[nodiscard]] inline Acc stencil_reduce_2d_(T const* data,
                                            std::array<std::size_t, 2> const& L,
                                            std::array<std::size_t, 2> const& stride,
                                            std::array<std::size_t, 2> const& lo,
                                            std::array<std::size_t, 2> const& hi,
                                            Comb const& comb,
                                            Body const& body) noexcept {
    std::size_t const L0 = L[0];
    std::size_t const L1 = L[1];
    std::size_t const s1 = stride[1];
    Acc total{};
    for (std::size_t y = lo[1]; y < hi[1]; ++y) {
        std::size_t const yp     = (y + 1 == L1) ? 0 : (y + 1);
        std::size_t const ym     = (y == 0) ? (L1 - 1) : (y - 1);
        std::size_t const row    = y * s1;
        std::size_t const row_yp = yp * s1;
        std::size_t const row_ym = ym * s1;
        x_sweep_<Policy>(L0, [&](std::size_t x, std::size_t xm, std::size_t xp) {
            T const self = data[row + x];
            if constexpr (Policy::fwd && Policy::bwd) {
                total += body(
                    self, nbr_sum(comb, data, self, row + xp, row + xm, row_yp + x, row_ym + x));
            } else if constexpr (Policy::fwd) {
                total += body(self, nbr_sum(comb, data, self, row + xp, row_yp + x));
            } else {
                total += body(self, nbr_sum(comb, data, self, row + xm, row_ym + x));
            }
        });
    }
    return total;
}

template <class Policy, class T, class Comb, class Body>
inline void stencil_map_3d_(T const* data,
                            std::array<std::size_t, 3> const& L,
                            std::array<std::size_t, 3> const& stride,
                            std::array<std::size_t, 3> const& lo,
                            std::array<std::size_t, 3> const& hi,
                            Comb const& comb,
                            Body const& body) noexcept {
    std::size_t const L0 = L[0];
    std::size_t const L1 = L[1];
    std::size_t const L2 = L[2];
    std::size_t const s1 = stride[1];
    std::size_t const s2 = stride[2];
    for (std::size_t z = lo[2]; z < hi[2]; ++z) {
        std::size_t const zp       = (z + 1 == L2) ? 0 : (z + 1);
        std::size_t const zm       = (z == 0) ? (L2 - 1) : (z - 1);
        std::size_t const plane    = z * s2;
        std::size_t const plane_zp = zp * s2;
        std::size_t const plane_zm = zm * s2;
        for (std::size_t y = lo[1]; y < hi[1]; ++y) {
            std::size_t const yp     = (y + 1 == L1) ? 0 : (y + 1);
            std::size_t const ym     = (y == 0) ? (L1 - 1) : (y - 1);
            std::size_t const row    = plane + (y * s1);
            std::size_t const row_yp = plane + (yp * s1);
            std::size_t const row_ym = plane + (ym * s1);
            std::size_t const row_zp = plane_zp + (y * s1);
            std::size_t const row_zm = plane_zm + (y * s1);
            x_sweep_<Policy>(L0, [&](std::size_t x, std::size_t xm, std::size_t xp) {
                std::size_t const i = row + x;
                T const self        = data[i];
                if constexpr (Policy::fwd && Policy::bwd) {
                    body(i,
                         self,
                         nbr_sum(comb,
                                 data,
                                 self,
                                 row + xp,
                                 row + xm,
                                 row_yp + x,
                                 row_ym + x,
                                 row_zp + x,
                                 row_zm + x));
                } else if constexpr (Policy::fwd) {
                    body(i, self, nbr_sum(comb, data, self, row + xp, row_yp + x, row_zp + x));
                } else {
                    body(i, self, nbr_sum(comb, data, self, row + xm, row_ym + x, row_zm + x));
                }
            });
        }
    }
}

template <class Policy, class Acc, class T, class Comb, class Body>
[[nodiscard]] inline Acc stencil_reduce_3d_(T const* data,
                                            std::array<std::size_t, 3> const& L,
                                            std::array<std::size_t, 3> const& stride,
                                            std::array<std::size_t, 3> const& lo,
                                            std::array<std::size_t, 3> const& hi,
                                            Comb const& comb,
                                            Body const& body) noexcept {
    std::size_t const L0 = L[0];
    std::size_t const L1 = L[1];
    std::size_t const L2 = L[2];
    std::size_t const s1 = stride[1];
    std::size_t const s2 = stride[2];
    Acc total{};
    for (std::size_t z = lo[2]; z < hi[2]; ++z) {
        std::size_t const zp       = (z + 1 == L2) ? 0 : (z + 1);
        std::size_t const zm       = (z == 0) ? (L2 - 1) : (z - 1);
        std::size_t const plane    = z * s2;
        std::size_t const plane_zp = zp * s2;
        std::size_t const plane_zm = zm * s2;
        for (std::size_t y = lo[1]; y < hi[1]; ++y) {
            std::size_t const yp     = (y + 1 == L1) ? 0 : (y + 1);
            std::size_t const ym     = (y == 0) ? (L1 - 1) : (y - 1);
            std::size_t const row    = plane + (y * s1);
            std::size_t const row_yp = plane + (yp * s1);
            std::size_t const row_ym = plane + (ym * s1);
            std::size_t const row_zp = plane_zp + (y * s1);
            std::size_t const row_zm = plane_zm + (y * s1);
            x_sweep_<Policy>(L0, [&](std::size_t x, std::size_t xm, std::size_t xp) {
                T const self = data[row + x];
                if constexpr (Policy::fwd && Policy::bwd) {
                    total += body(self,
                                  nbr_sum(comb,
                                          data,
                                          self,
                                          row + xp,
                                          row + xm,
                                          row_yp + x,
                                          row_ym + x,
                                          row_zp + x,
                                          row_zm + x));
                } else if constexpr (Policy::fwd) {
                    total +=
                        body(self, nbr_sum(comb, data, self, row + xp, row_yp + x, row_zp + x));
                } else {
                    total +=
                        body(self, nbr_sum(comb, data, self, row + xm, row_ym + x, row_zm + x));
                }
            });
        }
    }
    return total;
}

template <class Policy, class T, class Comb, class Body>
inline void stencil_map_4d_(T const* data,
                            std::array<std::size_t, 4> const& L,
                            std::array<std::size_t, 4> const& stride,
                            std::array<std::size_t, 4> const& lo,
                            std::array<std::size_t, 4> const& hi,
                            Comb const& comb,
                            Body const& body) noexcept {
    std::size_t const L0 = L[0];
    std::size_t const L1 = L[1];
    std::size_t const L2 = L[2];
    std::size_t const L3 = L[3];
    std::size_t const s1 = stride[1];
    std::size_t const s2 = stride[2];
    std::size_t const s3 = stride[3];
    for (std::size_t w = lo[3]; w < hi[3]; ++w) {
        std::size_t const wp     = (w + 1 == L3) ? 0 : (w + 1);
        std::size_t const wm     = (w == 0) ? (L3 - 1) : (w - 1);
        std::size_t const hyp    = w * s3;
        std::size_t const hyp_wp = wp * s3;
        std::size_t const hyp_wm = wm * s3;
        for (std::size_t z = lo[2]; z < hi[2]; ++z) {
            std::size_t const zp       = (z + 1 == L2) ? 0 : (z + 1);
            std::size_t const zm       = (z == 0) ? (L2 - 1) : (z - 1);
            std::size_t const plane    = hyp + (z * s2);
            std::size_t const plane_zp = hyp + (zp * s2);
            std::size_t const plane_zm = hyp + (zm * s2);
            std::size_t const plane_wp = hyp_wp + (z * s2);
            std::size_t const plane_wm = hyp_wm + (z * s2);
            for (std::size_t y = lo[1]; y < hi[1]; ++y) {
                std::size_t const yp     = (y + 1 == L1) ? 0 : (y + 1);
                std::size_t const ym     = (y == 0) ? (L1 - 1) : (y - 1);
                std::size_t const row    = plane + (y * s1);
                std::size_t const row_yp = plane + (yp * s1);
                std::size_t const row_ym = plane + (ym * s1);
                std::size_t const row_zp = plane_zp + (y * s1);
                std::size_t const row_zm = plane_zm + (y * s1);
                std::size_t const row_wp = plane_wp + (y * s1);
                std::size_t const row_wm = plane_wm + (y * s1);
                x_sweep_<Policy>(L0, [&](std::size_t x, std::size_t xm, std::size_t xp) {
                    std::size_t const i = row + x;
                    T const self        = data[i];
                    if constexpr (Policy::fwd && Policy::bwd) {
                        body(i,
                             self,
                             nbr_sum(comb,
                                     data,
                                     self,
                                     row + xp,
                                     row + xm,
                                     row_yp + x,
                                     row_ym + x,
                                     row_zp + x,
                                     row_zm + x,
                                     row_wp + x,
                                     row_wm + x));
                    } else if constexpr (Policy::fwd) {
                        body(i,
                             self,
                             nbr_sum(
                                 comb, data, self, row + xp, row_yp + x, row_zp + x, row_wp + x));
                    } else {
                        body(i,
                             self,
                             nbr_sum(
                                 comb, data, self, row + xm, row_ym + x, row_zm + x, row_wm + x));
                    }
                });
            }
        }
    }
}

template <class Policy, class Acc, class T, class Comb, class Body>
[[nodiscard]] inline Acc stencil_reduce_4d_(T const* data,
                                            std::array<std::size_t, 4> const& L,
                                            std::array<std::size_t, 4> const& stride,
                                            std::array<std::size_t, 4> const& lo,
                                            std::array<std::size_t, 4> const& hi,
                                            Comb const& comb,
                                            Body const& body) noexcept {
    std::size_t const L0 = L[0];
    std::size_t const L1 = L[1];
    std::size_t const L2 = L[2];
    std::size_t const L3 = L[3];
    std::size_t const s1 = stride[1];
    std::size_t const s2 = stride[2];
    std::size_t const s3 = stride[3];
    Acc total{};
    for (std::size_t w = lo[3]; w < hi[3]; ++w) {
        std::size_t const wp     = (w + 1 == L3) ? 0 : (w + 1);
        std::size_t const wm     = (w == 0) ? (L3 - 1) : (w - 1);
        std::size_t const hyp    = w * s3;
        std::size_t const hyp_wp = wp * s3;
        std::size_t const hyp_wm = wm * s3;
        for (std::size_t z = lo[2]; z < hi[2]; ++z) {
            std::size_t const zp       = (z + 1 == L2) ? 0 : (z + 1);
            std::size_t const zm       = (z == 0) ? (L2 - 1) : (z - 1);
            std::size_t const plane    = hyp + (z * s2);
            std::size_t const plane_zp = hyp + (zp * s2);
            std::size_t const plane_zm = hyp + (zm * s2);
            std::size_t const plane_wp = hyp_wp + (z * s2);
            std::size_t const plane_wm = hyp_wm + (z * s2);
            for (std::size_t y = lo[1]; y < hi[1]; ++y) {
                std::size_t const yp     = (y + 1 == L1) ? 0 : (y + 1);
                std::size_t const ym     = (y == 0) ? (L1 - 1) : (y - 1);
                std::size_t const row    = plane + (y * s1);
                std::size_t const row_yp = plane + (yp * s1);
                std::size_t const row_ym = plane + (ym * s1);
                std::size_t const row_zp = plane_zp + (y * s1);
                std::size_t const row_zm = plane_zm + (y * s1);
                std::size_t const row_wp = plane_wp + (y * s1);
                std::size_t const row_wm = plane_wm + (y * s1);
                x_sweep_<Policy>(L0, [&](std::size_t x, std::size_t xm, std::size_t xp) {
                    T const self = data[row + x];
                    if constexpr (Policy::fwd && Policy::bwd) {
                        total += body(self,
                                      nbr_sum(comb,
                                              data,
                                              self,
                                              row + xp,
                                              row + xm,
                                              row_yp + x,
                                              row_ym + x,
                                              row_zp + x,
                                              row_zm + x,
                                              row_wp + x,
                                              row_wm + x));
                    } else if constexpr (Policy::fwd) {
                        total += body(
                            self,
                            nbr_sum(
                                comb, data, self, row + xp, row_yp + x, row_zp + x, row_wp + x));
                    } else {
                        total += body(
                            self,
                            nbr_sum(
                                comb, data, self, row + xm, row_ym + x, row_zm + x, row_wm + x));
                    }
                });
            }
        }
    }
    return total;
}

// Map body(i, self, agg) over every site (agg folds the Policy-selected neighbours
// via `comb`). Write-disjoint → bit-identical for any partition / thread count.
template <class Policy, class T, class Comb, class Body>
inline void visit_stencil(Lattice<T> const& l, Comb const& comb, Body&& body) noexcept {
    Body const& b = body;
    traverse_dispatch_<void>(
        l,
        [&]<std::size_t D>(auto const& L, auto const& stride, auto const& lo, auto const& hi) {
            if constexpr (D == 2) {
                stencil_map_2d_<Policy>(l.data(), L, stride, lo, hi, comb, b);
            } else if constexpr (D == 3) {
                stencil_map_3d_<Policy>(l.data(), L, stride, lo, hi, comb, b);
            } else {
                stencil_map_4d_<Policy>(l.data(), L, stride, lo, hi, comb, b);
            }
        },
        [&](std::size_t x0, std::size_t cnt) {
            stencil_map_1d_<Policy>(l.data(), x0, x0 + cnt, l.shape()[0], comb, b);
        },
        [&](std::size_t s0, std::size_t cnt) {
            stencil_map_flat_<Policy, T>(l, s0, cnt, comb, b);
        });
}

// Reduce body(self, agg) over every site (agg folds the Policy-selected neighbours
// via `comb`). Partition summed in canonical item order → deterministic for a fixed
// (team, slabs) config; a different team re-folds.
template <class Policy, class T, class Acc = T, class Comb, class Body>
[[nodiscard]] inline Acc
reduce_stencil(Lattice<T> const& l, Comb const& comb, Body&& body) noexcept {
    Body const& b = body;
    return traverse_dispatch_<Acc>(
        l,
        [&]<std::size_t D>(auto const& L, auto const& stride, auto const& lo, auto const& hi) {
            if constexpr (D == 2) {
                return stencil_reduce_2d_<Policy, Acc>(l.data(), L, stride, lo, hi, comb, b);
            } else if constexpr (D == 3) {
                return stencil_reduce_3d_<Policy, Acc>(l.data(), L, stride, lo, hi, comb, b);
            } else {
                return stencil_reduce_4d_<Policy, Acc>(l.data(), L, stride, lo, hi, comb, b);
            }
        },
        [&](std::size_t x0, std::size_t cnt) {
            return stencil_reduce_1d_<Policy, Acc>(l.data(), x0, x0 + cnt, l.shape()[0], comb, b);
        },
        [&](std::size_t s0, std::size_t cnt) {
            return stencil_reduce_flat_<Policy, Acc, T>(l, s0, cnt, comb, b);
        });
}

}  // namespace reticolo::action::sweep
