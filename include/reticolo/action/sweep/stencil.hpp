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
//   visit_stencil(l, comb, body):    map. body(i, self, agg) -> void; the body
//                                    owns the write (out[i]=… or m[i]+=…). `agg`
//                                    folds all 2·ndims neighbours.
//   reduce_stencil(l, comb, body):   reduce. body(self, agg) -> Acc, accumulated
//                                    into a thread-count-invariant total. `agg`
//                                    folds the ndims FORWARD neighbours only (each
//                                    bond once).
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
// the gather fallback, any thread count); the reduce folds the fixed
// machine-independent partition in canonical item order (thread-count invariant
// AND identical across platforms).
//
// D in {1, 2, 3, 4} take the vectorised generic path; D > 4 falls back to a flat
// gather through the neighbour table (exact, just slower). Raising the vectorised
// ceiling is a one-line switch case — the stencil body already handles any D.
//
// Compile with -DRETICOLO_HOT_LOOP_FORCE_FALLBACK=1 to force every call onto the
// gather fallback regardless of ndims (the "old hot loop" bench path).

namespace reticolo::action::sweep {

// ---------- neighbour policy + combine ---------------------------------------

// The two neighbour sets, as compile-time tags. `all` selects both ±mu
// neighbours (map / force); otherwise only the +mu neighbours (reduce / s_full).
struct AllDirs {
    static constexpr bool all = true;
};
struct FwdOnly {
    static constexpr bool all = false;
};

// The site combine: fold raw neighbour values (self ignored). Inlines away so the
// site path is exactly `agg += data[nbr]` — bit-identical to the pre-generic loop.
struct IdentityCombine {
    template <class T>
    [[gnu::always_inline]] T operator()(T /*self*/, T nbr) const noexcept {
        return nbr;
    }
};

// ---------- gather fallback (any dimension) ----------------------------------

// Flat neighbour-table sweep over sites [s0, s0+cnt). Neighbour order
// (next[mu] then prev[mu], mu ascending) is the canonical order the vectorised
// engine reproduces bit-for-bit for IdentityCombine.
template <class T, class Comb, class Body>
inline void stencil_map_flat_(Lattice<T> const& l,
                              std::size_t s0,
                              std::size_t cnt,
                              Comb const& comb,
                              Body const& body) noexcept {
    auto const& idx              = l.indexing_ref();
    T const* data                = l.data();
    Site::value_type const* next = idx.next_data();
    Site::value_type const* prev = idx.prev_data();
    std::size_t const d          = idx.ndims();
    std::size_t const end        = s0 + cnt;

    for (std::size_t i = s0; i < end; ++i) {
        T const self           = data[i];
        T agg                  = T{0};
        std::size_t const base = i * d;
        for (std::size_t mu = 0; mu < d; ++mu) {
            agg += comb(self, data[next[base + mu]]);
            agg += comb(self, data[prev[base + mu]]);
        }
        body(i, self, agg);
    }
}

template <class T, class Acc, class Comb, class Body>
[[nodiscard]] inline Acc stencil_reduce_flat_(Lattice<T> const& l,
                                              std::size_t s0,
                                              std::size_t cnt,
                                              Comb const& comb,
                                              Body const& body) noexcept {
    RETICOLO_FP_REASSOCIATE
    auto const& idx              = l.indexing_ref();
    T const* data                = l.data();
    Site::value_type const* next = idx.next_data();
    std::size_t const d          = idx.ndims();
    std::size_t const end        = s0 + cnt;

    Acc total{};
    for (std::size_t i = s0; i < end; ++i) {
        T const self           = data[i];
        T agg                  = T{0};
        std::size_t const base = i * d;
        for (std::size_t mu = 0; mu < d; ++mu) {
            agg += comb(self, data[next[base + mu]]);
        }
        total += body(self, agg);
    }
    return total;
}

// Identity-combine fallbacks, kept under their historical names for the site unit
// tests (which use them as the bit-exact reference) and the site D>4 path.
template <class T, class Body>
inline void visit_nn_fallback_(Lattice<T> const& l,
                               std::size_t s0,
                               std::size_t cnt,
                               Body const& body) noexcept {
    stencil_map_flat_<T>(l, s0, cnt, IdentityCombine{}, body);
}

template <class T, class Acc = T, class Body>
[[nodiscard]] inline Acc reduce_fwd_fallback_(Lattice<T> const& l,
                                              std::size_t s0,
                                              std::size_t cnt,
                                              Body const& body) noexcept {
    return stencil_reduce_flat_<T, Acc>(l, s0, cnt, IdentityCombine{}, body);
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
inline Acc run_items_(bool want, std::size_t n_items, Work const& work) {
    if constexpr (std::is_void_v<Acc>) {
        reticolo::exec::parallel_map(want, n_items, work);
    } else {
        return reticolo::exec::parallel_reduce<Acc>(want, n_items, work);
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
inline Acc run_partition_items_(Lattice<T> const& l, bool want, Item const& item) {
    auto const [L, stride]  = geometry_<D>(l);
    exec::Partition const p = reticolo::exec::partition(l);
    return run_items_<Acc>(want, p.n_items, [&](std::size_t it) {
        std::array<std::size_t, D> lo{};
        std::array<std::size_t, D> hi{};
        for (std::size_t mu = 0; mu < D; ++mu) {
            hi[mu] = L[mu];
        }
        std::size_t rem = it;
        for (std::size_t mu = p.split_dim; mu < D; ++mu) {
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
    std::size_t const n              = l.nsites();
    std::size_t const bps            = l.bytes_per_site();
    [[maybe_unused]] bool const want = reticolo::exec::want_threads(n, bps);
#if RETICOLO_HOT_LOOP_FORCE_FALLBACK
    return run_ranges_<Acc>(n, bps, 1, flat);
#else
    switch (l.ndims()) {
        case 1:
            return run_ranges_<Acc>(l.shape()[0], bps, 1, one_d);
        case 2:
            return run_partition_items_<Acc, 2>(l, want, item);
        case 3:
            return run_partition_items_<Acc, 3>(l, want, item);
        case 4:
            return run_partition_items_<Acc, 4>(l, want, item);
        default:
            return run_ranges_<Acc>(n, bps, 1, flat);
    }
#endif
}

// ---------- hand-written per-dimension item nests ----------------------------
//
// One explicit hoisted loop nest per D over a partition item [lo,hi) (inner dim 0
// kept full). Each outer level hoists its neighbour row/plane/hypercube bases as
// scalars; the innermost dim-0 loop sums the neighbours INLINE (self at data[i],
// the ±x contiguous, the outer bases at `base + x`) and calls `body`. Map sums
// both ±mu (order x⁺,x⁻,y⁺,y⁻,…) and peels the two x-wrap ends; reduce sums the
// forward mu only (x⁺,y⁺,…) into `total` and peels just the +x wrap. Left-to-right
// association is the exact gather-fallback fold order → bit-identical for any
// thread count. Wraps use the full L[mu]. Writing the sum inline (not via a shared
// row helper) is what lets the compiler collapse the whole per-site kernel.

// 1D: a chunk [x0, xe) of the single axis. Only the two global x-wraps are peeled
// (a chunk that doesn't touch them runs branch-free). Map sums x⁺,x⁻; reduce sums
// x⁺ only. Order matches the gather fallback.
template <class T, class Comb, class Body>
inline void
stencil_map_1d_(T const* data, std::size_t x0, std::size_t xe, std::size_t L0, Comb const& comb,
             Body const& body) noexcept {
    std::size_t xb   = x0;
    std::size_t xend = xe;
    if (x0 == 0) {
        T const self = data[0];
        body(std::size_t{0}, self, comb(self, data[1]) + comb(self, data[L0 - 1]));
        xb = 1;
    }
    if (xe == L0) {
        xend = L0 - 1;
    }
    for (std::size_t x = xb; x < xend; ++x) {
        T const self = data[x];
        body(x, self, comb(self, data[x + 1]) + comb(self, data[x - 1]));
    }
    if (xe == L0) {
        std::size_t const i = L0 - 1;
        T const self        = data[i];
        body(i, self, comb(self, data[0]) + comb(self, data[i - 1]));
    }
}

template <class Acc, class T, class Comb, class Body>
[[nodiscard]] inline Acc
stencil_reduce_1d_(T const* data, std::size_t x0, std::size_t xe, std::size_t L0, Comb const& comb,
                Body const& body) noexcept {
    Acc total{};
    std::size_t const xend = (xe == L0) ? (L0 - 1) : xe;
    for (std::size_t x = x0; x < xend; ++x) {
        T const self = data[x];
        total += body(self, comb(self, data[x + 1]));
    }
    if (xe == L0) {
        std::size_t const i = L0 - 1;
        T const self        = data[i];
        total += body(self, comb(self, data[0]));
    }
    return total;
}

template <class T, class Comb, class Body>
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
        auto const sum = [&](std::size_t i, std::size_t xm, std::size_t xp, std::size_t off) {
            T const self = data[i];
            T const agg  = comb(self, data[xp]) + comb(self, data[xm]) +
                          comb(self, data[row_yp + off]) + comb(self, data[row_ym + off]);
            body(i, self, agg);
        };
        sum(row, row + (L0 - 1), row + 1, 0);
        for (std::size_t x = 1; x + 1 < L0; ++x) {
            sum(row + x, row + x - 1, row + x + 1, x);
        }
        sum(row + (L0 - 1), row + (L0 - 2), row, L0 - 1);
    }
}

template <class Acc, class T, class Comb, class Body>
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
        std::size_t const row    = y * s1;
        std::size_t const row_yp = yp * s1;
        for (std::size_t x = 0; x + 1 < L0; ++x) {
            std::size_t const i = row + x;
            T const self        = data[i];
            total += body(self, comb(self, data[i + 1]) + comb(self, data[row_yp + x]));
        }
        std::size_t const i = row + (L0 - 1);
        T const self        = data[i];
        total += body(self, comb(self, data[row]) + comb(self, data[row_yp + (L0 - 1)]));
    }
    return total;
}

template <class T, class Comb, class Body>
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
            auto const sum = [&](std::size_t i, std::size_t xm, std::size_t xp, std::size_t off) {
                T const self = data[i];
                T const agg  = comb(self, data[xp]) + comb(self, data[xm]) +
                              comb(self, data[row_yp + off]) + comb(self, data[row_ym + off]) +
                              comb(self, data[row_zp + off]) + comb(self, data[row_zm + off]);
                body(i, self, agg);
            };
            sum(row, row + (L0 - 1), row + 1, 0);
            for (std::size_t x = 1; x + 1 < L0; ++x) {
                sum(row + x, row + x - 1, row + x + 1, x);
            }
            sum(row + (L0 - 1), row + (L0 - 2), row, L0 - 1);
        }
    }
}

template <class Acc, class T, class Comb, class Body>
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
        std::size_t const plane    = z * s2;
        std::size_t const plane_zp = zp * s2;
        for (std::size_t y = lo[1]; y < hi[1]; ++y) {
            std::size_t const yp     = (y + 1 == L1) ? 0 : (y + 1);
            std::size_t const row    = plane + (y * s1);
            std::size_t const row_yp = plane + (yp * s1);
            std::size_t const row_zp = plane_zp + (y * s1);
            for (std::size_t x = 0; x + 1 < L0; ++x) {
                std::size_t const i = row + x;
                T const self        = data[i];
                total += body(self, comb(self, data[i + 1]) + comb(self, data[row_yp + x]) +
                                        comb(self, data[row_zp + x]));
            }
            std::size_t const i = row + (L0 - 1);
            T const self        = data[i];
            total += body(self, comb(self, data[row]) + comb(self, data[row_yp + (L0 - 1)]) +
                                    comb(self, data[row_zp + (L0 - 1)]));
        }
    }
    return total;
}

template <class T, class Comb, class Body>
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
                auto const sum = [&](std::size_t i, std::size_t xm, std::size_t xp,
                                     std::size_t off) {
                    T const self = data[i];
                    T const agg  = comb(self, data[xp]) + comb(self, data[xm]) +
                                  comb(self, data[row_yp + off]) + comb(self, data[row_ym + off]) +
                                  comb(self, data[row_zp + off]) + comb(self, data[row_zm + off]) +
                                  comb(self, data[row_wp + off]) + comb(self, data[row_wm + off]);
                    body(i, self, agg);
                };
                sum(row, row + (L0 - 1), row + 1, 0);
                for (std::size_t x = 1; x + 1 < L0; ++x) {
                    sum(row + x, row + x - 1, row + x + 1, x);
                }
                sum(row + (L0 - 1), row + (L0 - 2), row, L0 - 1);
            }
        }
    }
}

template <class Acc, class T, class Comb, class Body>
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
        std::size_t const hyp    = w * s3;
        std::size_t const hyp_wp = wp * s3;
        for (std::size_t z = lo[2]; z < hi[2]; ++z) {
            std::size_t const zp       = (z + 1 == L2) ? 0 : (z + 1);
            std::size_t const plane    = hyp + (z * s2);
            std::size_t const plane_zp = hyp + (zp * s2);
            std::size_t const plane_wp = hyp_wp + (z * s2);
            for (std::size_t y = lo[1]; y < hi[1]; ++y) {
                std::size_t const yp     = (y + 1 == L1) ? 0 : (y + 1);
                std::size_t const row    = plane + (y * s1);
                std::size_t const row_yp = plane + (yp * s1);
                std::size_t const row_zp = plane_zp + (y * s1);
                std::size_t const row_wp = plane_wp + (y * s1);
                for (std::size_t x = 0; x + 1 < L0; ++x) {
                    std::size_t const i = row + x;
                    T const self        = data[i];
                    total += body(self, comb(self, data[i + 1]) + comb(self, data[row_yp + x]) +
                                            comb(self, data[row_zp + x]) + comb(self, data[row_wp + x]));
                }
                std::size_t const i = row + (L0 - 1);
                T const self        = data[i];
                total += body(self, comb(self, data[row]) + comb(self, data[row_yp + (L0 - 1)]) +
                                        comb(self, data[row_zp + (L0 - 1)]) +
                                        comb(self, data[row_wp + (L0 - 1)]));
            }
        }
    }
    return total;
}

// Map body(i, self, agg) over every site (agg folds all 2·ndims neighbours via
// `comb`). Write-disjoint → bit-identical for any partition / thread count.
template <class T, class Comb, class Body>
inline void visit_stencil(Lattice<T> const& l, Comb const& comb, Body&& body) noexcept {
    Body const& b = body;
    traverse_dispatch_<void>(
        l,
        [&]<std::size_t D>(auto const& L, auto const& stride, auto const& lo, auto const& hi) {
            if constexpr (D == 2) {
                stencil_map_2d_(l.data(), L, stride, lo, hi, comb, b);
            } else if constexpr (D == 3) {
                stencil_map_3d_(l.data(), L, stride, lo, hi, comb, b);
            } else {
                stencil_map_4d_(l.data(), L, stride, lo, hi, comb, b);
            }
        },
        [&](std::size_t x0, std::size_t cnt) {
            stencil_map_1d_(l.data(), x0, x0 + cnt, l.shape()[0], comb, b);
        },
        [&](std::size_t s0, std::size_t cnt) { stencil_map_flat_<T>(l, s0, cnt, comb, b); });
}

// Reduce body(self, agg) over every site (agg folds the ndims forward neighbours
// via `comb`, each bond once). Fixed work-item partition summed in canonical order
// → identical for any thread count.
template <class T, class Acc = T, class Comb, class Body>
[[nodiscard]] inline Acc
reduce_stencil(Lattice<T> const& l, Comb const& comb, Body&& body) noexcept {
    Body const& b = body;
    return traverse_dispatch_<Acc>(
        l,
        [&]<std::size_t D>(auto const& L, auto const& stride, auto const& lo, auto const& hi) {
            if constexpr (D == 2) {
                return stencil_reduce_2d_<Acc>(l.data(), L, stride, lo, hi, comb, b);
            } else if constexpr (D == 3) {
                return stencil_reduce_3d_<Acc>(l.data(), L, stride, lo, hi, comb, b);
            } else {
                return stencil_reduce_4d_<Acc>(l.data(), L, stride, lo, hi, comb, b);
            }
        },
        [&](std::size_t x0, std::size_t cnt) {
            return stencil_reduce_1d_<Acc>(l.data(), x0, x0 + cnt, l.shape()[0], comb, b);
        },
        [&](std::size_t s0, std::size_t cnt) {
            return stencil_reduce_flat_<T, Acc>(l, s0, cnt, comb, b);
        });
}

}  // namespace reticolo::action::sweep
