#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/parallel.hpp>
#include <reticolo/core/site.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
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

// Dimension-generic nearest-neighbour traversal engine, shared by every NN scalar
// family (site, bond, complex). One engine sweeps the lattice as a stack of
// innermost-axis rows: dim 0 (x) is kept FULL and contiguous so the inner loop
// vectorises on every architecture (NEON / SSE / AVX2 / AVX-512 / SVE) with no
// intrinsics; the outer axes (dims 1..D-1) are enumerated by `walk_outer_<D>`, a
// compile-time recursive loop nest that the compiler expands — for a fixed D —
// into the identical hand-written D-deep nest. So the per-dim stencil is written
// once, not once per dimension.
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
// Partition (work-item granularity) is chosen per dimension: 1D chunks the inner
// axis; 2D takes one row per item; 3D one plane; 4D one cache tile. The map output
// is write-disjoint (bit-identical to the gather fallback, any thread count); the
// reduce folds a deterministic fixed partition in canonical item order (thread-
// count invariant).
//
// D in {1, 2, 3, 4} take the vectorised generic path; D > 4 falls back to a flat
// gather through the neighbour table (exact, just slower). Raising the vectorised
// ceiling is a one-line switch case plus a `plan_block` root — the stencil body
// already handles any D.
//
// Compile with -DRETICOLO_HOT_LOOP_FORCE_FALLBACK=1 to force every call onto the
// gather fallback regardless of ndims (the "old hot loop" bench path).

namespace reticolo::action::detail {

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

// ---------- generic stencil body ---------------------------------------------

// Per-site neighbour aggregate, in the fallback-matching order
// (inner-fwd, inner-bwd, then for each outer dim fwd, bwd). `fwd`/`bwd` hold the
// pre-wrapped neighbour ROW bases for the outer dims (index 0 unused); the inner
// (dim-0) neighbour is the contiguous data[i±1]. `comb(self, ·)` is applied per
// neighbour. Three variants: the contiguous interior (no wrap) and the two peeled
// global-x edges. All `always_inline` so the compile-time-bounded `for mu`
// unrolls and the interior vectorises.

template <std::size_t D, class Policy, class T, class Comb>
[[gnu::always_inline]] inline T agg_bulk_(T const* data,
                                          T self,
                                          std::size_t i,
                                          std::size_t x,
                                          std::array<std::size_t, D> const& fwd,
                                          std::array<std::size_t, D> const& bwd,
                                          Comb const& comb) noexcept {
    T a = comb(self, data[i + 1]);
    if constexpr (Policy::all) {
        a += comb(self, data[i - 1]);
    }
    for (std::size_t mu = 1; mu < D; ++mu) {
        a += comb(self, data[fwd[mu] + x]);
        if constexpr (Policy::all) {
            a += comb(self, data[bwd[mu] + x]);
        }
    }
    return a;
}

// Global -x edge (x == 0, i == own): inner +x is data[own+1], inner -x wraps to
// data[own+L0-1]. Outer neighbours read at x = 0.
template <std::size_t D, class Policy, class T, class Comb>
[[gnu::always_inline]] inline T agg_lo_(T const* data,
                                        T self,
                                        std::size_t own,
                                        std::size_t L0,
                                        std::array<std::size_t, D> const& fwd,
                                        std::array<std::size_t, D> const& bwd,
                                        Comb const& comb) noexcept {
    T a = comb(self, data[own + 1]);
    if constexpr (Policy::all) {
        a += comb(self, data[own + (L0 - 1)]);
    }
    for (std::size_t mu = 1; mu < D; ++mu) {
        a += comb(self, data[fwd[mu]]);
        if constexpr (Policy::all) {
            a += comb(self, data[bwd[mu]]);
        }
    }
    return a;
}

// Global +x edge (x == L0-1, i == own+L0-1): inner +x wraps to data[own], inner
// -x is data[i-1]. Outer neighbours read at x = L0-1.
template <std::size_t D, class Policy, class T, class Comb>
[[gnu::always_inline]] inline T agg_hi_(T const* data,
                                        T self,
                                        std::size_t own,
                                        std::size_t L0,
                                        std::array<std::size_t, D> const& fwd,
                                        std::array<std::size_t, D> const& bwd,
                                        Comb const& comb) noexcept {
    std::size_t const i = own + (L0 - 1);
    T a                 = comb(self, data[own]);
    if constexpr (Policy::all) {
        a += comb(self, data[i - 1]);
    }
    for (std::size_t mu = 1; mu < D; ++mu) {
        a += comb(self, data[fwd[mu] + (L0 - 1)]);
        if constexpr (Policy::all) {
            a += comb(self, data[bwd[mu] + (L0 - 1)]);
        }
    }
    return a;
}

// Map one innermost row: sites [x0, x1) at row base `own`. The two global-x wraps
// are peeled off; the interior [xb, xe) is the contiguous vectorised run. Any
// chunk partition of a full row reproduces the whole-row sweep bit-for-bit (only
// the global edges wrap; internal block faces read the contiguous parent).
template <std::size_t D, class Policy, class T, class Comb, class Body>
inline void map_row_(T const* data,
                     std::size_t own,
                     std::array<std::size_t, D> const& fwd,
                     std::array<std::size_t, D> const& bwd,
                     std::size_t x0,
                     std::size_t x1,
                     std::size_t L0,
                     Comb const& comb,
                     Body const& body) noexcept {
    std::size_t xb = x0;
    std::size_t xe = x1;
    if (x0 == 0) {
        body(own, data[own], agg_lo_<D, Policy>(data, data[own], own, L0, fwd, bwd, comb));
        xb = 1;
    }
    if (x1 == L0) {
        xe = L0 - 1;
    }
    for (std::size_t x = xb; x < xe; ++x) {
        std::size_t const i = own + x;
        body(i, data[i], agg_bulk_<D, Policy>(data, data[i], i, x, fwd, bwd, comb));
    }
    if (x1 == L0) {
        std::size_t const i = own + (L0 - 1);
        body(i, data[i], agg_hi_<D, Policy>(data, data[i], own, L0, fwd, bwd, comb));
    }
}

// Reduce one innermost row: same peeled structure, folding body(self, agg) into a
// per-row partial. FwdOnly only, so the x==0 peel produces the identical value as
// the interior (no -x term) — kept uniform for one code path.
template <std::size_t D, class Policy, class Acc, class T, class Comb, class Body>
[[nodiscard]] inline Acc reduce_row_(T const* data,
                                     std::size_t own,
                                     std::array<std::size_t, D> const& fwd,
                                     std::array<std::size_t, D> const& bwd,
                                     std::size_t x0,
                                     std::size_t x1,
                                     std::size_t L0,
                                     Comb const& comb,
                                     Body const& body) noexcept {
    RETICOLO_FP_REASSOCIATE
    Acc total{};
    std::size_t xb = x0;
    std::size_t xe = x1;
    if (x0 == 0) {
        total += body(data[own], agg_lo_<D, Policy>(data, data[own], own, L0, fwd, bwd, comb));
        xb = 1;
    }
    if (x1 == L0) {
        xe = L0 - 1;
    }
    for (std::size_t x = xb; x < xe; ++x) {
        std::size_t const i = own + x;
        total += body(data[i], agg_bulk_<D, Policy>(data, data[i], i, x, fwd, bwd, comb));
    }
    if (x1 == L0) {
        std::size_t const i = own + (L0 - 1);
        total += body(data[i], agg_hi_<D, Policy>(data, data[i], own, L0, fwd, bwd, comb));
    }
    return total;
}

// Compile-time recursive loop nest over the outer dims [1, D). `Mu` counts down
// from D-1 (outermost) to 0; at Mu==0 all outer coords are fixed and `emit(own,
// coord)` is called for one innermost row. For a fixed D this expands into the
// identical `for w { for z { for y { ... } } }` nest written by hand before — but
// as ONE definition covering every dimension. `own` accumulates the row base
// during descent; `coord` records the outer coordinates for the neighbour-base
// computation at the bottom.
template <std::size_t D, std::size_t Mu, class T, class Emit>
inline void walk_outer_(std::array<std::size_t, D> const& stride,
                        std::array<std::size_t, D> const& lo,
                        std::array<std::size_t, D> const& hi,
                        std::size_t own,
                        std::array<std::size_t, D>& coord,
                        Emit const& emit) noexcept {
    if constexpr (Mu == 0) {
        emit(own, coord);
    } else {
        for (std::size_t c = lo[Mu]; c < hi[Mu]; ++c) {
            coord[Mu] = c;
            walk_outer_<D, Mu - 1, T>(stride, lo, hi, own + (c * stride[Mu]), coord, emit);
        }
    }
}

// Run one work item (a rectangular sub-lattice [lo, hi), inner dim full) as a map.
// The emit lambda derives each row's pre-wrapped outer neighbour bases from `own`
// and `coord`, then maps the row.
template <std::size_t D, class Policy, class T, class Comb, class Body>
inline void map_item_(Lattice<T> const& l,
                      std::array<std::size_t, D> const& L,
                      std::array<std::size_t, D> const& stride,
                      std::array<std::size_t, D> const& lo,
                      std::array<std::size_t, D> const& hi,
                      Comb const& comb,
                      Body const& body) noexcept {
    T const* const data  = l.data();
    std::size_t const L0 = L[0];
    std::array<std::size_t, D> coord{};
    walk_outer_<D, D - 1, T>(
        stride, lo, hi, std::size_t{0}, coord, [&](std::size_t own, auto const& cc) {
            std::array<std::size_t, D> fwd{};
            std::array<std::size_t, D> bwd{};
            for (std::size_t nu = 1; nu < D; ++nu) {
                std::size_t const c   = cc[nu];
                std::size_t const bnu = own - (c * stride[nu]);
                std::size_t const cf  = (c + 1 == L[nu]) ? std::size_t{0} : (c + 1);
                std::size_t const cb  = (c == 0) ? (L[nu] - 1) : (c - 1);
                fwd[nu]               = bnu + (cf * stride[nu]);
                bwd[nu]               = bnu + (cb * stride[nu]);
            }
            map_row_<D, Policy>(data, own, fwd, bwd, std::size_t{0}, L0, L0, comb, body);
        });
}

// Run one work item as a reduce; per-row partials fold in odometer (walk_outer_)
// order → the item partial is thread-count-independent.
template <std::size_t D, class Policy, class Acc, class T, class Comb, class Body>
[[nodiscard]] inline Acc reduce_item_(Lattice<T> const& l,
                                      std::array<std::size_t, D> const& L,
                                      std::array<std::size_t, D> const& stride,
                                      std::array<std::size_t, D> const& lo,
                                      std::array<std::size_t, D> const& hi,
                                      Comb const& comb,
                                      Body const& body) noexcept {
    RETICOLO_FP_REASSOCIATE
    T const* const data  = l.data();
    std::size_t const L0 = L[0];
    Acc total{};
    std::array<std::size_t, D> coord{};
    walk_outer_<D, D - 1, T>(
        stride, lo, hi, std::size_t{0}, coord, [&](std::size_t own, auto const& cc) {
            std::array<std::size_t, D> fwd{};
            std::array<std::size_t, D> bwd{};
            for (std::size_t nu = 1; nu < D; ++nu) {
                std::size_t const c   = cc[nu];
                std::size_t const bnu = own - (c * stride[nu]);
                std::size_t const cf  = (c + 1 == L[nu]) ? std::size_t{0} : (c + 1);
                std::size_t const cb  = (c == 0) ? (L[nu] - 1) : (c - 1);
                fwd[nu]               = bnu + (cf * stride[nu]);
                bwd[nu]               = bnu + (cb * stride[nu]);
            }
            total += reduce_row_<D, Policy, Acc>(
                data, own, fwd, bwd, std::size_t{0}, L0, L0, comb, body);
        });
    return total;
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

// ---------- cache-tile plan (4D) ---------------------------------------------

// The sub-lattice is a slab: FULL in the innermost dimension (x), blocked in
// y,z,w. This is deliberate — a stencil's x-neighbour has reuse distance 1 (the
// adjacent element, always cached), so tiling x cannot cut DRAM traffic and only
// shortens the vectorised inner run. The dims that overflow cache are y,z,w
// (reuse distances L0, L0·L1, L0·L1·L2), so those are the ones blocked, with side
// b from the halo working set L0·(b+2)³·sizeof(T) ≤ target. Each block is a
// self-contained parallel work unit. (Measured: tiling x is 1.5–5× slower, purely
// from SIMD starvation, even single-threaded with cache to spare.)
struct Block4d {
    std::size_t bx, by, bz, bw;
};

template <class T>
[[nodiscard]] inline Block4d
plan_block_4d_(std::size_t L0, std::size_t L1, std::size_t L2, std::size_t L3) noexcept {
    // x stays FULL — the innermost axis carries the contiguous memory streams the
    // hardware prefetcher relies on, so tiling it breaks those streams and costs
    // far more than the halo it saves. Block y,z,w so the halo working set
    // L0·(b+2)³·sizeof(T) fits the fixed per-core target (see k_traverse_l2_bytes).
    double const cap = static_cast<double>(reticolo::detail::k_traverse_l2_bytes) /
                       (static_cast<double>(L0) * static_cast<double>(sizeof(T)));
    long b = static_cast<long>(std::cbrt(cap)) - 2;  // L0·(b+2)³ fits the budget
    if (b < 4) {
        b = 4;
    }
    auto const ub = static_cast<std::size_t>(b);
    return {L0, std::min(ub, L1), std::min(ub, L2), std::min(ub, L3)};
}

// ---------- dispatch ---------------------------------------------------------

// Map body(i, self, agg) over every site (agg folds all 2·ndims neighbours via
// `comb`). Write-disjoint → bit-identical for any partition / thread count.
template <class T, class Comb, class Body>
inline void visit_stencil(Lattice<T> const& l, Comb const& comb, Body&& body) noexcept {
    std::size_t const n   = l.nsites();
    std::size_t const bps = l.bytes_per_site();
    bool const want       = reticolo::detail::want_threads(n, bps);
    Body const& b         = body;
#if RETICOLO_HOT_LOOP_FORCE_FALLBACK
    reticolo::detail::parallel_map_ranges(n, bps, 1, [&](std::size_t s0, std::size_t cnt) {
        stencil_map_flat_<T>(l, s0, cnt, comb, b);
    });
#else
    switch (l.ndims()) {
        case 1: {
            std::size_t const L0 = l.shape()[0];
            T const* const data  = l.data();
            std::array<std::size_t, 1> const nb{};  // no outer dims
            reticolo::detail::parallel_map_ranges(L0, bps, 1, [&](std::size_t x0, std::size_t cnt) {
                map_row_<1, AllDirs>(data, 0, nb, nb, x0, x0 + cnt, L0, comb, b);
            });
            return;
        }
        case 2: {
            auto const [L, stride] = geometry_<2>(l);
            reticolo::detail::parallel_map(want, L[1], [&](std::size_t y) {
                std::array<std::size_t, 2> const lo{0, y};
                std::array<std::size_t, 2> const hi{L[0], y + 1};
                map_item_<2, AllDirs>(l, L, stride, lo, hi, comb, b);
            });
            return;
        }
        case 3: {
            auto const [L, stride] = geometry_<3>(l);
            reticolo::detail::parallel_map(want, L[2], [&](std::size_t z) {
                std::array<std::size_t, 3> const lo{0, 0, z};
                std::array<std::size_t, 3> const hi{L[0], L[1], z + 1};
                map_item_<3, AllDirs>(l, L, stride, lo, hi, comb, b);
            });
            return;
        }
        case 4: {
            auto const [L, stride] = geometry_<4>(l);
            Block4d const bl       = plan_block_4d_<T>(L[0], L[1], L[2], L[3]);
            std::size_t const nty  = (L[1] + bl.by - 1) / bl.by;
            std::size_t const ntz  = (L[2] + bl.bz - 1) / bl.bz;
            std::size_t const ntw  = (L[3] + bl.bw - 1) / bl.bw;
            reticolo::detail::parallel_map(want, nty * ntz * ntw, [&](std::size_t item) {
                std::size_t const ty = item % nty;
                std::size_t const r  = item / nty;
                std::size_t const tz = r % ntz;
                std::size_t const tw = r / ntz;
                std::array<std::size_t, 4> const lo{0, ty * bl.by, tz * bl.bz, tw * bl.bw};
                std::array<std::size_t, 4> const hi{L[0],
                                                    std::min(lo[1] + bl.by, L[1]),
                                                    std::min(lo[2] + bl.bz, L[2]),
                                                    std::min(lo[3] + bl.bw, L[3])};
                map_item_<4, AllDirs>(l, L, stride, lo, hi, comb, b);
            });
            return;
        }
        default:
            reticolo::detail::parallel_map_ranges(n, bps, 1, [&](std::size_t s0, std::size_t cnt) {
                stencil_map_flat_<T>(l, s0, cnt, comb, b);
            });
            return;
    }
#endif
}

// Reduce body(self, agg) over every site (agg folds the ndims forward neighbours
// via `comb`, each bond once). Fixed work-item partition summed in canonical order
// → identical for any thread count.
template <class T, class Acc = T, class Comb, class Body>
[[nodiscard]] inline Acc
reduce_stencil(Lattice<T> const& l, Comb const& comb, Body&& body) noexcept {
    std::size_t const n   = l.nsites();
    std::size_t const bps = l.bytes_per_site();
    bool const want       = reticolo::detail::want_threads(n, bps);
    Body const& b         = body;
#if RETICOLO_HOT_LOOP_FORCE_FALLBACK
    return reticolo::detail::parallel_reduce_ranges<Acc>(
        n, bps, 1, [&](std::size_t s0, std::size_t cnt) {
            return stencil_reduce_flat_<T, Acc>(l, s0, cnt, comb, b);
        });
#else
    switch (l.ndims()) {
        case 1: {
            std::size_t const L0 = l.shape()[0];
            T const* const data  = l.data();
            std::array<std::size_t, 1> const nb{};
            return reticolo::detail::parallel_reduce_ranges<Acc>(
                L0, bps, 1, [&](std::size_t x0, std::size_t cnt) {
                    return reduce_row_<1, FwdOnly, Acc>(data, 0, nb, nb, x0, x0 + cnt, L0, comb, b);
                });
        }
        case 2: {
            auto const [L, stride] = geometry_<2>(l);
            return reticolo::detail::parallel_reduce<Acc>(want, L[1], [&](std::size_t y) {
                std::array<std::size_t, 2> const lo{0, y};
                std::array<std::size_t, 2> const hi{L[0], y + 1};
                return reduce_item_<2, FwdOnly, Acc>(l, L, stride, lo, hi, comb, b);
            });
        }
        case 3: {
            auto const [L, stride] = geometry_<3>(l);
            return reticolo::detail::parallel_reduce<Acc>(want, L[2], [&](std::size_t z) {
                std::array<std::size_t, 3> const lo{0, 0, z};
                std::array<std::size_t, 3> const hi{L[0], L[1], z + 1};
                return reduce_item_<3, FwdOnly, Acc>(l, L, stride, lo, hi, comb, b);
            });
        }
        case 4: {
            auto const [L, stride] = geometry_<4>(l);
            Block4d const bl       = plan_block_4d_<T>(L[0], L[1], L[2], L[3]);
            std::size_t const nty  = (L[1] + bl.by - 1) / bl.by;
            std::size_t const ntz  = (L[2] + bl.bz - 1) / bl.bz;
            std::size_t const ntw  = (L[3] + bl.bw - 1) / bl.bw;
            return reticolo::detail::parallel_reduce<Acc>(
                want, nty * ntz * ntw, [&](std::size_t item) {
                    std::size_t const ty = item % nty;
                    std::size_t const r  = item / nty;
                    std::size_t const tz = r % ntz;
                    std::size_t const tw = r / ntz;
                    std::array<std::size_t, 4> const lo{0, ty * bl.by, tz * bl.bz, tw * bl.bw};
                    std::array<std::size_t, 4> const hi{L[0],
                                                        std::min(lo[1] + bl.by, L[1]),
                                                        std::min(lo[2] + bl.bz, L[2]),
                                                        std::min(lo[3] + bl.bw, L[3])};
                    return reduce_item_<4, FwdOnly, Acc>(l, L, stride, lo, hi, comb, b);
                });
        }
        default:
            return reticolo::detail::parallel_reduce_ranges<Acc>(
                n, bps, 1, [&](std::size_t s0, std::size_t cnt) {
                    return stencil_reduce_flat_<T, Acc>(l, s0, cnt, comb, b);
                });
    }
#endif
}

}  // namespace reticolo::action::detail
