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

// Chunk for the flat-iteration paths (1D and the D>4 fallback), in sites. The
// row/plane/tile paths use their own natural work items instead.
inline constexpr std::size_t k_site_chunk = 1UL << 13;  // 8192 sites

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

// Flat neighbour-table sweep over sites [s0, s0+cnt). Any dimension; the general
// per-item worker for D>4 (and the forced-fallback bench path).
template <class T, class Body>
inline void visit_nn_fallback_(Lattice<T> const& l,
                               std::size_t s0,
                               std::size_t cnt,
                               Body const& body) noexcept {
    auto const& idx              = l.indexing_ref();
    T const* data                = l.data();
    Site::value_type const* next = idx.next_data();
    Site::value_type const* prev = idx.prev_data();
    std::size_t const d          = idx.ndims();
    std::size_t const end        = s0 + cnt;

    for (std::size_t i = s0; i < end; ++i) {
        T nbrs                 = T{0};
        std::size_t const base = i * d;
        for (std::size_t mu = 0; mu < d; ++mu) {
            nbrs += data[next[base + mu]];
            nbrs += data[prev[base + mu]];
        }
        body(i, data[i], nbrs);
    }
}

// x-range [x0, x1) of a 1D lattice. Interior x reads data[x±1]; the global x=0 /
// x=Lx-1 wraps are peeled off when they fall in range, so any chunk partition is
// bit-identical to the full sweep. Order: next[0] + prev[0] (matches fallback).
template <class T, class Body>
inline void
visit_nn_1d_(Lattice<T> const& l, std::size_t x0, std::size_t x1, Body const& body) noexcept {
    T const* data        = l.data();
    std::size_t const Lx = l.shape()[0];
    std::size_t xb       = x0;
    std::size_t xe       = x1;
    if (x0 == 0) {  // global -x wrap
        body(std::size_t{0}, data[0], data[1] + data[Lx - 1]);
        xb = 1;
    }
    if (x1 == Lx) {
        xe = Lx - 1;
    }
    for (std::size_t x = xb; x < xe; ++x) {
        body(x, data[x], data[x + 1] + data[x - 1]);
    }
    if (x1 == Lx) {  // global +x wrap
        body(Lx - 1, data[Lx - 1], data[0] + data[Lx - 2]);
    }
}

// Rows [y0, y1) of a 2D lattice (inner x full). Each row is a self-contained
// work unit — the parallel map hands one y-range to each thread.
template <class T, class Body>
inline void
visit_nn_2d_(Lattice<T> const& l, std::size_t y0, std::size_t y1, Body const& body) noexcept {
    T const* data        = l.data();
    auto const& sh       = l.shape();
    std::size_t const Lx = sh[0];
    std::size_t const Ly = sh[1];
    std::size_t const sy = Lx;

    for (std::size_t y = y0; y < y1; ++y) {
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

// Planes [z0, z1) of a 3D lattice (inner y,x full). One z-range per work item.
template <class T, class Body>
inline void
visit_nn_3d_(Lattice<T> const& l, std::size_t z0, std::size_t z1, Body const& body) noexcept {
    T const* data        = l.data();
    auto const& sh       = l.shape();
    std::size_t const Lx = sh[0];
    std::size_t const Ly = sh[1];
    std::size_t const Lz = sh[2];
    std::size_t const sy = Lx;
    std::size_t const sz = Lx * Ly;

    for (std::size_t z = z0; z < z1; ++z) {
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

// 4D handler: work items = tiles of the y/z/w grid (x full — see plan_block_4d_).
// The cache tiling caps each work unit's halo working set; `parallel_map` splits
// the tile grid across the team. Force output is bit-identical to the serial
// sweep (each site written once, order-independent). The serial branch (!want or
// foreign region) walks all tiles on one thread = the full sweep.
template <class T, class Body>
inline void visit_nn_tiled_4d_(Lattice<T> const& l, bool want, Body const& body) noexcept {
    auto const& sh        = l.shape();
    std::size_t const L0  = sh[0];
    std::size_t const L1  = sh[1];
    std::size_t const L2  = sh[2];
    std::size_t const L3  = sh[3];
    Block4d const bl      = plan_block_4d_<T>(L0, L1, L2, L3);
    std::size_t const ntx = (L0 + bl.bx - 1) / bl.bx;
    std::size_t const nty = (L1 + bl.by - 1) / bl.by;
    std::size_t const ntz = (L2 + bl.bz - 1) / bl.bz;
    std::size_t const ntw = (L3 + bl.bw - 1) / bl.bw;

    reticolo::detail::parallel_map(want, ntx * nty * ntz * ntw, [&](std::size_t item) {
        std::size_t const tx = item % ntx;
        std::size_t r        = item / ntx;
        std::size_t const ty = r % nty;
        r /= nty;
        std::size_t const tz = r % ntz;
        std::size_t const tw = r / ntz;
        std::size_t const x0 = tx * bl.bx;
        std::size_t const y0 = ty * bl.by;
        std::size_t const z0 = tz * bl.bz;
        std::size_t const w0 = tw * bl.bw;
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
    });
}

// Dispatch by dimension. Every dimension threads through the two primitives: 1D
// and the D>4 fallback over site chunks, 2D over rows, 3D over planes, 4D over
// cache tiles. The per-dim inner kernel stays the hand-written vectorised loop.
template <class T, class Body>
inline void visit_nn(Lattice<T> const& l, Body&& body) noexcept {
    std::size_t const n = l.nsites();
    bool const want     = reticolo::detail::traverse_want(n);
    Body const& b       = body;
#if RETICOLO_HOT_LOOP_FORCE_FALLBACK
    reticolo::detail::parallel_map_ranges(
        want, n, k_site_chunk, [&](std::size_t s0, std::size_t cnt) {
            visit_nn_fallback_(l, s0, cnt, b);
        });
#else
    auto const& sh = l.shape();
    switch (l.ndims()) {
        case 1:
            reticolo::detail::parallel_map_ranges(
                want, sh[0], k_site_chunk, [&](std::size_t x0, std::size_t cnt) {
                    visit_nn_1d_(l, x0, x0 + cnt, b);
                });
            return;
        case 2:
            reticolo::detail::parallel_map(
                want, sh[1], [&](std::size_t y) { visit_nn_2d_(l, y, y + 1, b); });
            return;
        case 3:
            reticolo::detail::parallel_map(
                want, sh[2], [&](std::size_t z) { visit_nn_3d_(l, z, z + 1, b); });
            return;
        case 4:
            visit_nn_tiled_4d_(l, want, b);
            return;
        default:
            reticolo::detail::parallel_map_ranges(
                want, n, k_site_chunk, [&](std::size_t s0, std::size_t cnt) {
                    visit_nn_fallback_(l, s0, cnt, b);
                });
            return;
    }
#endif
}

// ---------- reduce_fwd -------------------------------------------------------

template <class T, class Acc = T, class Body>
[[nodiscard]] inline Acc reduce_fwd_fallback_(Lattice<T> const& l,
                                              std::size_t s0,
                                              std::size_t cnt,
                                              Body const& body) noexcept {
    RETICOLO_FP_REASSOCIATE
    auto const& idx              = l.indexing_ref();
    T const* data                = l.data();
    Site::value_type const* next = idx.next_data();
    std::size_t const d          = idx.ndims();
    std::size_t const end        = s0 + cnt;

    Acc total{};
    for (std::size_t i = s0; i < end; ++i) {
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
[[nodiscard]] inline Acc
reduce_fwd_1d_(Lattice<T> const& l, std::size_t x0, std::size_t x1, Body const& body) noexcept {
    RETICOLO_FP_REASSOCIATE
    T const* data        = l.data();
    std::size_t const Lx = l.shape()[0];
    std::size_t const xe = (x1 == Lx) ? (Lx - 1) : x1;
    Acc total{};
    for (std::size_t x = x0; x < xe; ++x) {
        total += body(data[x], data[x + 1]);
    }
    if (x1 == Lx) {  // global +x wrap
        total += body(data[Lx - 1], data[0]);
    }
    return total;
}

template <class T, class Acc = T, class Body>
[[nodiscard]] inline Acc
reduce_fwd_2d_(Lattice<T> const& l, std::size_t y0, std::size_t y1, Body const& body) noexcept {
    RETICOLO_FP_REASSOCIATE
    T const* data        = l.data();
    auto const& sh       = l.shape();
    std::size_t const Lx = sh[0];
    std::size_t const Ly = sh[1];
    std::size_t const sy = Lx;
    Acc total{};
    for (std::size_t y = y0; y < y1; ++y) {
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
[[nodiscard]] inline Acc
reduce_fwd_3d_(Lattice<T> const& l, std::size_t z0, std::size_t z1, Body const& body) noexcept {
    RETICOLO_FP_REASSOCIATE
    T const* data        = l.data();
    auto const& sh       = l.shape();
    std::size_t const Lx = sh[0];
    std::size_t const Ly = sh[1];
    std::size_t const Lz = sh[2];
    std::size_t const sy = Lx;
    std::size_t const sz = Lx * Ly;
    Acc total{};
    for (std::size_t z = z0; z < z1; ++z) {
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
        total +=
            body(data[i], data[i + 1] + data[row_yp + x] + data[row_zp + x] + data[row_wp + x]);
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
// 4D s_full reduction: work items = tiles of the same fixed grid as the force map,
// so the result is thread-count invariant (each tile's partial is order-fixed,
// partials folded in canonical tile order). NB: the tile grouping changes the
// summation order vs a single running sum — a one-time bit re-baseline of s_full.
template <class T, class Acc = T, class Body>
[[nodiscard]] inline Acc reduce_fwd_4d_(Lattice<T> const& l, bool want, Body const& body) noexcept {
    auto const& sh        = l.shape();
    std::size_t const L0  = sh[0];
    std::size_t const L1  = sh[1];
    std::size_t const L2  = sh[2];
    std::size_t const L3  = sh[3];
    Block4d const bl      = plan_block_4d_<T>(L0, L1, L2, L3);
    std::size_t const ntx = (L0 + bl.bx - 1) / bl.bx;
    std::size_t const nty = (L1 + bl.by - 1) / bl.by;
    std::size_t const ntz = (L2 + bl.bz - 1) / bl.bz;
    std::size_t const ntw = (L3 + bl.bw - 1) / bl.bw;

    return reticolo::detail::parallel_reduce<Acc>(
        want, ntx * nty * ntz * ntw, [&](std::size_t item) {
            std::size_t const tx = item % ntx;
            std::size_t r        = item / ntx;
            std::size_t const ty = r % nty;
            r /= nty;
            std::size_t const tz = r % ntz;
            std::size_t const tw = r / ntz;
            std::size_t const x0 = tx * bl.bx;
            std::size_t const y0 = ty * bl.by;
            std::size_t const z0 = tz * bl.bz;
            std::size_t const w0 = tw * bl.bw;
            return reduce_block_fwd_4d_<T, Acc>(l,
                                                x0,
                                                std::min(x0 + bl.bx, L0),
                                                y0,
                                                std::min(y0 + bl.by, L1),
                                                z0,
                                                std::min(z0 + bl.bz, L2),
                                                w0,
                                                std::min(w0 + bl.bw, L3),
                                                body);
        });
}

template <class T, class Acc = T, class Body>
[[nodiscard]] inline Acc reduce_fwd(Lattice<T> const& l, Body&& body) noexcept {
    std::size_t const n = l.nsites();
    bool const want     = reticolo::detail::traverse_want(n);
    Body const& b       = body;
#if RETICOLO_HOT_LOOP_FORCE_FALLBACK
    return reticolo::detail::parallel_reduce_ranges<Acc>(
        want, n, k_site_chunk, [&](std::size_t s0, std::size_t cnt) {
            return reduce_fwd_fallback_<T, Acc>(l, s0, cnt, b);
        });
#else
    auto const& sh = l.shape();
    switch (l.ndims()) {
        case 1:
            return reticolo::detail::parallel_reduce_ranges<Acc>(
                want, sh[0], k_site_chunk, [&](std::size_t x0, std::size_t cnt) {
                    return reduce_fwd_1d_<T, Acc>(l, x0, x0 + cnt, b);
                });
        case 2:
            return reticolo::detail::parallel_reduce<Acc>(
                want, sh[1], [&](std::size_t y) { return reduce_fwd_2d_<T, Acc>(l, y, y + 1, b); });
        case 3:
            return reticolo::detail::parallel_reduce<Acc>(
                want, sh[2], [&](std::size_t z) { return reduce_fwd_3d_<T, Acc>(l, z, z + 1, b); });
        case 4:
            return reduce_fwd_4d_<T, Acc>(l, want, b);
        default:
            return reticolo::detail::parallel_reduce_ranges<Acc>(
                want, n, k_site_chunk, [&](std::size_t s0, std::size_t cnt) {
                    return reduce_fwd_fallback_<T, Acc>(l, s0, cnt, b);
                });
    }
#endif
}

}  // namespace reticolo::action::detail
