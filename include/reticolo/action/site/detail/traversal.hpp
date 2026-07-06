#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/parallel.hpp>
#include <reticolo/core/site.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

// Function-scope FP-reassociation hint for reduction loops. On clang this
// breaks the strict left-to-right `total += body(i)` dependency chain so
// independent body evaluations (sin/cos/heavy fma) can overlap. Other
// compilers still build; they just lose this particular micro-opt.
#if defined(__clang__)
    #define RETICOLO_FP_REASSOCIATE _Pragma("clang fp reassociate(on)")
#else
    #define RETICOLO_FP_REASSOCIATE
#endif

namespace reticolo::action::detail {

// Hot-loop helpers for scalar nearest-neighbour kernels on periodic hypercubic
// lattices. Two patterns:
//
//  visit_nn(l, body):    body(i, phi, nbrs_sum) -> void.
//                        nbrs_sum is the unweighted sum of all 2*ndims nearest
//                        neighbours of site i. Body owns the write (e.g.
//                        out[i] = ... or m[i] += ...). Used by compute_force
//                        and the fused force+kick kernel.
//
//  reduce_fwd(l, body):  body(phi, fwd_sum) -> T. fwd_sum is the sum of the d
//                        positive-mu neighbours only (each bond counted once);
//                        helper accumulates the returned per-site value into a
//                        scalar total. Used by s_full.
//
// For ndims in {1, 2, 3, 4}, the inner-most axis loop is split into pure
// stride-arithmetic ranges plus an O(1) wrap correction at each end. The
// remaining axes' wraps are hoisted out of the inner loop as loop-invariant
// offsets. Compilers autovectorise the inner-x range on every architecture
// (NEON / SSE / AVX2 / AVX-512 / SVE) — no intrinsics, no architecture
// switches.
//
// ndims > 4 falls back to a flat gather through the neighbour table — exact,
// just slower. Adding ndim=5/6 is a mechanical extension.
//
// Compile with -DRETICOLO_HOT_LOOP_FORCE_FALLBACK=1 to force every call onto
// the gather fallback regardless of ndims. This is the "old hot loop" path
// used by `bench_scalars` to produce before/after numbers.

// ---------- visit_nn ---------------------------------------------------------

template <class T, class Body>
inline void visit_nn_fallback_(Lattice<T> const& l, Body&& body) noexcept {
    auto const& idx              = l.indexing_ref();
    T const* data                = l.data();
    Site::value_type const* next = idx.next_data();
    Site::value_type const* prev = idx.prev_data();
    std::size_t const n          = idx.nsites();
    std::size_t const d          = idx.ndims();

    for (std::size_t i = 0; i < n; ++i) {
        T nbrs                 = T{0};
        std::size_t const base = i * d;
        for (std::size_t mu = 0; mu < d; ++mu) {
            nbrs += data[next[base + mu]];
            nbrs += data[prev[base + mu]];
        }
        body(i, data[i], nbrs);
    }
}

template <class T, class Body>
inline void visit_nn_1d_(Lattice<T> const& l, Body&& body) noexcept {
    T const* data        = l.data();
    std::size_t const Lx = l.shape()[0];

    // Order: next[0] + prev[0]  (matches gather fallback bit-for-bit).
    // x = 0
    body(std::size_t{0}, data[0], data[1] + data[Lx - 1]);
    // x in (0, Lx-1)
    for (std::size_t x = 1; x + 1 < Lx; ++x) {
        body(x, data[x], data[x + 1] + data[x - 1]);
    }
    // x = Lx-1
    body(Lx - 1, data[Lx - 1], data[0] + data[Lx - 2]);
}

template <class T, class Body>
inline void visit_nn_2d_(Lattice<T> const& l, Body&& body) noexcept {
    T const* data        = l.data();
    auto const& sh       = l.shape();
    std::size_t const Lx = sh[0];
    std::size_t const Ly = sh[1];
    std::size_t const sy = Lx;

    for (std::size_t y = 0; y < Ly; ++y) {
        std::size_t const ym     = (y == 0) ? (Ly - 1) : (y - 1);
        std::size_t const yp     = (y + 1 == Ly) ? 0 : (y + 1);
        std::size_t const row    = y * sy;
        std::size_t const row_ym = ym * sy;
        std::size_t const row_yp = yp * sy;

        // Order: next[0] + prev[0] + next[1] + prev[1] (matches gather fallback).
        // x = 0
        {
            std::size_t const i = row;
            body(i, data[i], data[i + 1] + data[row + (Lx - 1)] + data[row_yp] + data[row_ym]);
        }
        // x in (0, Lx-1) — bulk
        for (std::size_t x = 1; x + 1 < Lx; ++x) {
            std::size_t const i = row + x;
            body(i, data[i], data[i + 1] + data[i - 1] + data[row_yp + x] + data[row_ym + x]);
        }
        // x = Lx-1
        {
            std::size_t const i = row + (Lx - 1);
            body(i,
                 data[i],
                 data[row] + data[i - 1] + data[row_yp + (Lx - 1)] + data[row_ym + (Lx - 1)]);
        }
    }
}

template <class T, class Body>
inline void visit_nn_3d_(Lattice<T> const& l, Body&& body) noexcept {
    T const* data        = l.data();
    auto const& sh       = l.shape();
    std::size_t const Lx = sh[0];
    std::size_t const Ly = sh[1];
    std::size_t const Lz = sh[2];
    std::size_t const sy = Lx;
    std::size_t const sz = Lx * Ly;

    for (std::size_t z = 0; z < Lz; ++z) {
        std::size_t const zm       = (z == 0) ? (Lz - 1) : (z - 1);
        std::size_t const zp       = (z + 1 == Lz) ? 0 : (z + 1);
        std::size_t const plane    = z * sz;
        std::size_t const plane_zm = zm * sz;
        std::size_t const plane_zp = zp * sz;

        for (std::size_t y = 0; y < Ly; ++y) {
            std::size_t const ym     = (y == 0) ? (Ly - 1) : (y - 1);
            std::size_t const yp     = (y + 1 == Ly) ? 0 : (y + 1);
            std::size_t const row    = plane + y * sy;
            std::size_t const row_ym = plane + ym * sy;
            std::size_t const row_yp = plane + yp * sy;
            std::size_t const row_zm = plane_zm + y * sy;
            std::size_t const row_zp = plane_zp + y * sy;

            // Order: next[0]+prev[0]+next[1]+prev[1]+next[2]+prev[2].
            // x = 0
            {
                std::size_t const i = row;
                body(i,
                     data[i],
                     data[i + 1] + data[row + (Lx - 1)] + data[row_yp] + data[row_ym] +
                         data[row_zp] + data[row_zm]);
            }
            // x in (0, Lx-1) — bulk, vectorizable
            for (std::size_t x = 1; x + 1 < Lx; ++x) {
                std::size_t const i = row + x;
                body(i,
                     data[i],
                     data[i + 1] + data[i - 1] + data[row_yp + x] + data[row_ym + x] +
                         data[row_zp + x] + data[row_zm + x]);
            }
            // x = Lx-1
            {
                std::size_t const i = row + (Lx - 1);
                body(i,
                     data[i],
                     data[row] + data[i - 1] + data[row_yp + (Lx - 1)] + data[row_ym + (Lx - 1)] +
                         data[row_zp + (Lx - 1)] + data[row_zm + (Lx - 1)]);
            }
        }
    }
}

// One row segment [x0,x1) of a 4D sub-lattice (fixed w,z,y). Interior x uses
// stride-1 reads `data[i±1]` — correct at internal block faces too, since the
// parent array is contiguous in x; only the *global* x=0 / x=L0-1 edges wrap.
// y,z,w neighbour row bases are precomputed by the caller (global wrap). Shared
// by the serial and tiled sweeps so both stay bit-identical.
template <class T, class Body>
inline void emit_row_nn_4d_(T const* data,
                            std::size_t row,
                            std::size_t row_ym,
                            std::size_t row_yp,
                            std::size_t row_zm,
                            std::size_t row_zp,
                            std::size_t row_wm,
                            std::size_t row_wp,
                            std::size_t x0,
                            std::size_t x1,
                            std::size_t L0,
                            Body& body) noexcept {
    // Order: next[0]+prev[0]+next[1]+prev[1]+next[2]+prev[2]+next[3]+prev[3].
    std::size_t xb = x0;
    std::size_t xe = x1;
    if (x0 == 0) {  // global -x wrap peeled off
        std::size_t const i = row;
        body(i,
             data[i],
             data[i + 1] + data[row + (L0 - 1)] + data[row_yp] + data[row_ym] + data[row_zp] +
                 data[row_zm] + data[row_wp] + data[row_wm]);
        xb = 1;
    }
    if (x1 == L0) {
        xe = L0 - 1;
    }
    for (std::size_t x = xb; x < xe; ++x) {  // interior — vectorised, no wrap
        std::size_t const i = row + x;
        body(i,
             data[i],
             data[i + 1] + data[i - 1] + data[row_yp + x] + data[row_ym + x] + data[row_zp + x] +
                 data[row_zm + x] + data[row_wp + x] + data[row_wm + x]);
    }
    if (x1 == L0) {  // global +x wrap peeled off
        std::size_t const i = row + (L0 - 1);
        body(i,
             data[i],
             data[row] + data[i - 1] + data[row_yp + (L0 - 1)] + data[row_ym + (L0 - 1)] +
                 data[row_zp + (L0 - 1)] + data[row_zm + (L0 - 1)] + data[row_wp + (L0 - 1)] +
                 data[row_wm + (L0 - 1)]);
    }
}

// Sweep a rectangular sub-lattice (a "view" over the shared field) tiled in all
// four dimensions [x0,x1)×[y0,y1)×[z0,z1)×[w0,w1). An interior site's neighbours
// are the sub-lattice's own data; only sites on the sub-lattice surface reach
// into an adjacent block (the implicit halo — a plain strided read into the
// shared parent, no copy). Full bounds reproduce the pre-tiling sweep for x=0,
// x=L0-1 and bit-for-bit for the interior.
template <class T, class Body>
inline void visit_block_nn_4d_(Lattice<T> const& l,
                               std::size_t x0,
                               std::size_t x1,
                               std::size_t y0,
                               std::size_t y1,
                               std::size_t z0,
                               std::size_t z1,
                               std::size_t w0,
                               std::size_t w1,
                               Body& body) noexcept {
    T const* data        = l.data();
    auto const& sh       = l.shape();
    std::size_t const L0 = sh[0];
    std::size_t const L1 = sh[1];
    std::size_t const L2 = sh[2];
    std::size_t const L3 = sh[3];
    std::size_t const s1 = L0;
    std::size_t const s2 = L0 * L1;
    std::size_t const s3 = L0 * L1 * L2;

    for (std::size_t w = w0; w < w1; ++w) {
        std::size_t const wm     = (w == 0) ? (L3 - 1) : (w - 1);
        std::size_t const wp     = (w + 1 == L3) ? 0 : (w + 1);
        std::size_t const hyp    = w * s3;
        std::size_t const hyp_wm = wm * s3;
        std::size_t const hyp_wp = wp * s3;
        for (std::size_t z = z0; z < z1; ++z) {
            std::size_t const zm       = (z == 0) ? (L2 - 1) : (z - 1);
            std::size_t const zp       = (z + 1 == L2) ? 0 : (z + 1);
            std::size_t const plane    = hyp + z * s2;
            std::size_t const plane_zm = hyp + zm * s2;
            std::size_t const plane_zp = hyp + zp * s2;
            std::size_t const plane_wm = hyp_wm + z * s2;
            std::size_t const plane_wp = hyp_wp + z * s2;
            for (std::size_t y = y0; y < y1; ++y) {
                std::size_t const ym = (y == 0) ? (L1 - 1) : (y - 1);
                std::size_t const yp = (y + 1 == L1) ? 0 : (y + 1);
                emit_row_nn_4d_(data,
                                plane + y * s1,
                                plane + ym * s1,
                                plane + yp * s1,
                                plane_zm + y * s1,
                                plane_zp + y * s1,
                                plane_wm + y * s1,
                                plane_wp + y * s1,
                                x0,
                                x1,
                                L0,
                                body);
            }
        }
    }
}

// Cache-tile plan. The sub-lattice is a slab: FULL in the innermost dimension
// (x), blocked in y,z,w. This is deliberate — a stencil's x-neighbour has reuse
// distance 1 (the adjacent element, always cached), so tiling x cannot cut DRAM
// traffic and only shortens the vectorised inner run. The dims that overflow
// cache are y,z,w (reuse distances L0, L0·L1, L0·L1·L2), so those are the ones
// blocked, with side b from the halo working set L0·(b+2)³·sizeof(T) ≤ target.
// Each block is still a self-contained parallel work unit. The traversal below
// takes an x-range too, so this is a one-line change if a huge-L0 case ever
// needs x tiled — but for stencils, x stays full. (Measured: tiling x is 1.5–5×
// slower, purely from SIMD starvation, even single-threaded with cache to spare.)
struct Block4d {
    std::size_t bx, by, bz, bw;
};

template <class T>
[[nodiscard]] inline Block4d
plan_block_4d_(std::size_t L0, std::size_t L1, std::size_t L2, std::size_t L3) noexcept {
    // x stays FULL — the innermost axis carries the contiguous memory streams the
    // hardware prefetcher relies on, so tiling it breaks those streams and costs
    // far more than the halo it saves (measured: tiling x ~halved L=96 throughput,
    // even single-threaded). Block y,z,w so the halo working set L0·(b+2)³·sizeof(T)
    // fits the fixed per-core target (see k_traverse_l2_bytes — filling a detected
    // L2 was measured to overshoot the sweet spot). The small blocks give the
    // thread team plenty of independent tiles to balance across.
    double const cap = static_cast<double>(reticolo::detail::k_traverse_l2_bytes) /
                       (static_cast<double>(L0) * static_cast<double>(sizeof(T)));
    long b = static_cast<long>(std::cbrt(cap)) - 2;  // L0·(b+2)³ fits the budget
    if (b < 4) {
        b = 4;
    }
    auto const ub = static_cast<std::size_t>(b);
    return {L0, std::min(ub, L1), std::min(ub, L2), std::min(ub, L3)};
}

// Serial full-lattice sweep — one block spanning the whole lattice.
template <class T, class Body>
inline void visit_nn_4d_(Lattice<T> const& l, Body&& body) noexcept {
    auto const& sh = l.shape();
    visit_block_nn_4d_<T>(l, 0, sh[0], 0, sh[1], 0, sh[2], 0, sh[3], body);
}

// Threaded, all-dims-tiled sweep: one omp region, sub-lattices are the work
// units, collapse(4) flattens the tile grid so threads pull whole blocks. No
// per-call allocation. Force output is bit-identical to the serial sweep (each
// site is written exactly once, order-independent).
template <class T, class Body>
inline void visit_nn_tiled_4d_(Lattice<T> const& l, Body&& body) noexcept {
    auto const& sh       = l.shape();
    std::size_t const L0 = sh[0];
    std::size_t const L1 = sh[1];
    std::size_t const L2 = sh[2];
    std::size_t const L3 = sh[3];
    Block4d const bl     = plan_block_4d_<T>(L0, L1, L2, L3);
    auto const ntx       = static_cast<std::ptrdiff_t>((L0 + bl.bx - 1) / bl.bx);
    auto const nty       = static_cast<std::ptrdiff_t>((L1 + bl.by - 1) / bl.by);
    auto const ntz       = static_cast<std::ptrdiff_t>((L2 + bl.bz - 1) / bl.bz);
    auto const ntw       = static_cast<std::ptrdiff_t>((L3 + bl.bw - 1) / bl.bw);

#pragma omp parallel for collapse(4) schedule(static)
    for (std::ptrdiff_t tw = 0; tw < ntw; ++tw) {
        for (std::ptrdiff_t tz = 0; tz < ntz; ++tz) {
            for (std::ptrdiff_t ty = 0; ty < nty; ++ty) {
                for (std::ptrdiff_t tx = 0; tx < ntx; ++tx) {
                    std::size_t const x0 = static_cast<std::size_t>(tx) * bl.bx;
                    std::size_t const y0 = static_cast<std::size_t>(ty) * bl.by;
                    std::size_t const z0 = static_cast<std::size_t>(tz) * bl.bz;
                    std::size_t const w0 = static_cast<std::size_t>(tw) * bl.bw;
                    visit_block_nn_4d_<T>(l,
                                          x0,
                                          std::min(x0 + bl.bx, L0),
                                          y0,
                                          std::min(y0 + bl.by, L1),
                                          z0,
                                          std::min(z0 + bl.bz, L2),
                                          w0,
                                          std::min(w0 + bl.bw, L3),
                                          body);
                }
            }
        }
    }
}

template <class T, class Body>
inline void visit_nn(Lattice<T> const& l, Body&& body) noexcept {
#if RETICOLO_HOT_LOOP_FORCE_FALLBACK
    visit_nn_fallback_(l, std::forward<Body>(body));
#else
    switch (l.ndims()) {
        case 1:
            visit_nn_1d_(l, std::forward<Body>(body));
            return;
        case 2:
            visit_nn_2d_(l, std::forward<Body>(body));
            return;
        case 3:
            visit_nn_3d_(l, std::forward<Body>(body));
            return;
        case 4:
            if (reticolo::detail::traverse_parallel(l.nsites())) {
                visit_nn_tiled_4d_(l, std::forward<Body>(body));
            } else {
                visit_nn_4d_(l, std::forward<Body>(body));
            }
            return;
        default:
            visit_nn_fallback_(l, std::forward<Body>(body));
            return;
    }
#endif
}

// ---------- reduce_fwd -------------------------------------------------------

template <class T, class Acc = T, class Body>
[[nodiscard]] inline Acc reduce_fwd_fallback_(Lattice<T> const& l, Body&& body) noexcept {
    RETICOLO_FP_REASSOCIATE
    auto const& idx              = l.indexing_ref();
    T const* data                = l.data();
    Site::value_type const* next = idx.next_data();
    std::size_t const n          = idx.nsites();
    std::size_t const d          = idx.ndims();

    Acc total{};
    for (std::size_t i = 0; i < n; ++i) {
        T fwd_sum              = T{0};
        std::size_t const base = i * d;
        for (std::size_t mu = 0; mu < d; ++mu) {
            fwd_sum += data[next[base + mu]];
        }
        total += body(data[i], fwd_sum);
    }
    return total;
}

template <class T, class Acc = T, class Body>
[[nodiscard]] inline Acc reduce_fwd_1d_(Lattice<T> const& l, Body&& body) noexcept {
    RETICOLO_FP_REASSOCIATE
    T const* data        = l.data();
    std::size_t const Lx = l.shape()[0];
    Acc total{};
    for (std::size_t x = 0; x + 1 < Lx; ++x) {
        total += body(data[x], data[x + 1]);
    }
    total += body(data[Lx - 1], data[0]);
    return total;
}

template <class T, class Acc = T, class Body>
[[nodiscard]] inline Acc reduce_fwd_2d_(Lattice<T> const& l, Body&& body) noexcept {
    RETICOLO_FP_REASSOCIATE
    T const* data        = l.data();
    auto const& sh       = l.shape();
    std::size_t const Lx = sh[0];
    std::size_t const Ly = sh[1];
    std::size_t const sy = Lx;
    Acc total{};
    for (std::size_t y = 0; y < Ly; ++y) {
        std::size_t const yp     = (y + 1 == Ly) ? 0 : (y + 1);
        std::size_t const row    = y * sy;
        std::size_t const row_yp = yp * sy;
        for (std::size_t x = 0; x + 1 < Lx; ++x) {
            std::size_t const i = row + x;
            total += body(data[i], data[i + 1] + data[row_yp + x]);
        }
        std::size_t const i = row + (Lx - 1);
        total += body(data[i], data[row] + data[row_yp + (Lx - 1)]);
    }
    return total;
}

template <class T, class Acc = T, class Body>
[[nodiscard]] inline Acc reduce_fwd_3d_(Lattice<T> const& l, Body&& body) noexcept {
    RETICOLO_FP_REASSOCIATE
    T const* data        = l.data();
    auto const& sh       = l.shape();
    std::size_t const Lx = sh[0];
    std::size_t const Ly = sh[1];
    std::size_t const Lz = sh[2];
    std::size_t const sy = Lx;
    std::size_t const sz = Lx * Ly;
    Acc total{};
    for (std::size_t z = 0; z < Lz; ++z) {
        std::size_t const zp       = (z + 1 == Lz) ? 0 : (z + 1);
        std::size_t const plane    = z * sz;
        std::size_t const plane_zp = zp * sz;
        for (std::size_t y = 0; y < Ly; ++y) {
            std::size_t const yp     = (y + 1 == Ly) ? 0 : (y + 1);
            std::size_t const row    = plane + y * sy;
            std::size_t const row_yp = plane + yp * sy;
            std::size_t const row_zp = plane_zp + y * sy;
            for (std::size_t x = 0; x + 1 < Lx; ++x) {
                std::size_t const i = row + x;
                total += body(data[i], data[i + 1] + data[row_yp + x] + data[row_zp + x]);
            }
            std::size_t const i = row + (Lx - 1);
            total += body(data[i], data[row] + data[row_yp + (Lx - 1)] + data[row_zp + (Lx - 1)]);
        }
    }
    return total;
}

// Forward-sum of one 4D row segment [x0,x1) (fixed w,z,y). Forward neighbours
// (row_yp/zp/wp) carry the global periodic wrap; x's forward neighbour is the
// contiguous data[i+1] except at the global x=L0-1 edge. Each site in the
// segment is counted once, so tiles partition the bonds cleanly.
template <class T, class Acc, class Body>
[[nodiscard]] inline Acc reduce_row_fwd_4d_(T const* data,
                                            std::size_t row,
                                            std::size_t row_yp,
                                            std::size_t row_zp,
                                            std::size_t row_wp,
                                            std::size_t x0,
                                            std::size_t x1,
                                            std::size_t L0,
                                            Body& body) noexcept {
    RETICOLO_FP_REASSOCIATE
    Acc total{};
    std::size_t const xe = (x1 == L0) ? (L0 - 1) : x1;
    for (std::size_t x = x0; x < xe; ++x) {
        std::size_t const i = row + x;
        total += body(data[i], data[i + 1] + data[row_yp + x] + data[row_zp + x] + data[row_wp + x]);
    }
    if (x1 == L0) {  // global +x wrap
        std::size_t const i = row + (L0 - 1);
        total += body(data[i],
                      data[row] + data[row_yp + (L0 - 1)] + data[row_zp + (L0 - 1)] +
                          data[row_wp + (L0 - 1)]);
    }
    return total;
}

// Forward-sum over an all-dims sub-lattice [x0,x1)×[y0,y1)×[z0,z1)×[w0,w1). Own
// partial, so tiles reduce independently and the caller sums them in a fixed
// order (thread-count-invariant s_full).
template <class T, class Acc, class Body>
[[nodiscard]] inline Acc reduce_block_fwd_4d_(Lattice<T> const& l,
                                              std::size_t x0,
                                              std::size_t x1,
                                              std::size_t y0,
                                              std::size_t y1,
                                              std::size_t z0,
                                              std::size_t z1,
                                              std::size_t w0,
                                              std::size_t w1,
                                              Body& body) noexcept {
    RETICOLO_FP_REASSOCIATE
    T const* data        = l.data();
    auto const& sh       = l.shape();
    std::size_t const L0 = sh[0];
    std::size_t const L2 = sh[2];
    std::size_t const L3 = sh[3];
    std::size_t const s1 = L0;
    std::size_t const s2 = L0 * sh[1];
    std::size_t const s3 = s2 * L2;
    Acc total{};
    for (std::size_t w = w0; w < w1; ++w) {
        std::size_t const wp     = (w + 1 == L3) ? 0 : (w + 1);
        std::size_t const hyp    = w * s3;
        std::size_t const hyp_wp = wp * s3;
        for (std::size_t z = z0; z < z1; ++z) {
            std::size_t const zp       = (z + 1 == L2) ? 0 : (z + 1);
            std::size_t const plane    = hyp + z * s2;
            std::size_t const plane_zp = hyp + zp * s2;
            std::size_t const plane_wp = hyp_wp + z * s2;
            for (std::size_t y = y0; y < y1; ++y) {
                std::size_t const yp = (y + 1 == sh[1]) ? 0 : (y + 1);
                total += reduce_row_fwd_4d_<T, Acc>(data,
                                                    plane + y * s1,
                                                    plane + yp * s1,
                                                    plane_zp + y * s1,
                                                    plane_wp + y * s1,
                                                    x0,
                                                    x1,
                                                    L0,
                                                    body);
            }
        }
    }
    return total;
}

// s_full reduction: always decomposed into the same fixed sub-lattice grid, so
// the result is identical for any thread count (each tile's partial is
// order-fixed, partials summed in canonical tile order). Threaded only past the
// size/nesting gate. NB: the tile grouping changes the summation order vs the
// pre-tiling single running sum — a one-time bit-level re-baseline of s_full.
template <class T, class Acc = T, class Body>
[[nodiscard]] inline Acc reduce_fwd_4d_(Lattice<T> const& l, Body&& body) noexcept {
    auto const& sh       = l.shape();
    std::size_t const L0 = sh[0];
    std::size_t const L1 = sh[1];
    std::size_t const L2 = sh[2];
    std::size_t const L3 = sh[3];
    Block4d const bl     = plan_block_4d_<T>(L0, L1, L2, L3);
    auto const ntx       = (L0 + bl.bx - 1) / bl.bx;
    auto const nty       = (L1 + bl.by - 1) / bl.by;
    auto const ntz       = (L2 + bl.bz - 1) / bl.bz;
    auto const ntw       = (L3 + bl.bw - 1) / bl.bw;
    std::vector<Acc> partials(ntx * nty * ntz * ntw, Acc{});
    [[maybe_unused]] bool const par = reticolo::detail::traverse_parallel(l.nsites());

#pragma omp parallel for collapse(4) schedule(static) if (par)
    for (std::ptrdiff_t tw = 0; tw < static_cast<std::ptrdiff_t>(ntw); ++tw) {
        for (std::ptrdiff_t tz = 0; tz < static_cast<std::ptrdiff_t>(ntz); ++tz) {
            for (std::ptrdiff_t ty = 0; ty < static_cast<std::ptrdiff_t>(nty); ++ty) {
                for (std::ptrdiff_t tx = 0; tx < static_cast<std::ptrdiff_t>(ntx); ++tx) {
                    std::size_t const x0 = static_cast<std::size_t>(tx) * bl.bx;
                    std::size_t const y0 = static_cast<std::size_t>(ty) * bl.by;
                    std::size_t const z0 = static_cast<std::size_t>(tz) * bl.bz;
                    std::size_t const w0 = static_cast<std::size_t>(tw) * bl.bw;
                    std::size_t const idx =
                        ((static_cast<std::size_t>(tw) * ntz + static_cast<std::size_t>(tz)) * nty +
                         static_cast<std::size_t>(ty)) *
                            ntx +
                        static_cast<std::size_t>(tx);
                    partials[idx] = reduce_block_fwd_4d_<T, Acc>(l,
                                                                 x0,
                                                                 std::min(x0 + bl.bx, L0),
                                                                 y0,
                                                                 std::min(y0 + bl.by, L1),
                                                                 z0,
                                                                 std::min(z0 + bl.bz, L2),
                                                                 w0,
                                                                 std::min(w0 + bl.bw, L3),
                                                                 body);
                }
            }
        }
    }
    Acc total{};
    for (Acc const& p : partials) {
        total += p;
    }
    return total;
}

template <class T, class Acc = T, class Body>
[[nodiscard]] inline Acc reduce_fwd(Lattice<T> const& l, Body&& body) noexcept {
#if RETICOLO_HOT_LOOP_FORCE_FALLBACK
    return reduce_fwd_fallback_<T, Acc>(l, std::forward<Body>(body));
#else
    switch (l.ndims()) {
        case 1:
            return reduce_fwd_1d_<T, Acc>(l, std::forward<Body>(body));
        case 2:
            return reduce_fwd_2d_<T, Acc>(l, std::forward<Body>(body));
        case 3:
            return reduce_fwd_3d_<T, Acc>(l, std::forward<Body>(body));
        case 4:
            return reduce_fwd_4d_<T, Acc>(l, std::forward<Body>(body));
        default:
            return reduce_fwd_fallback_<T, Acc>(l, std::forward<Body>(body));
    }
#endif
}

}  // namespace reticolo::action::detail
