#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/site.hpp>

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

template <class T, class Body>
inline void visit_nn_4d_(Lattice<T> const& l, Body&& body) noexcept {
    T const* data        = l.data();
    auto const& sh       = l.shape();
    std::size_t const L0 = sh[0];
    std::size_t const L1 = sh[1];
    std::size_t const L2 = sh[2];
    std::size_t const L3 = sh[3];
    std::size_t const s1 = L0;
    std::size_t const s2 = L0 * L1;
    std::size_t const s3 = L0 * L1 * L2;

    for (std::size_t w = 0; w < L3; ++w) {
        std::size_t const wm     = (w == 0) ? (L3 - 1) : (w - 1);
        std::size_t const wp     = (w + 1 == L3) ? 0 : (w + 1);
        std::size_t const hyp    = w * s3;
        std::size_t const hyp_wm = wm * s3;
        std::size_t const hyp_wp = wp * s3;

        for (std::size_t z = 0; z < L2; ++z) {
            std::size_t const zm       = (z == 0) ? (L2 - 1) : (z - 1);
            std::size_t const zp       = (z + 1 == L2) ? 0 : (z + 1);
            std::size_t const plane    = hyp + z * s2;
            std::size_t const plane_zm = hyp + zm * s2;
            std::size_t const plane_zp = hyp + zp * s2;
            std::size_t const plane_wm = hyp_wm + z * s2;
            std::size_t const plane_wp = hyp_wp + z * s2;

            for (std::size_t y = 0; y < L1; ++y) {
                std::size_t const ym     = (y == 0) ? (L1 - 1) : (y - 1);
                std::size_t const yp     = (y + 1 == L1) ? 0 : (y + 1);
                std::size_t const row    = plane + y * s1;
                std::size_t const row_ym = plane + ym * s1;
                std::size_t const row_yp = plane + yp * s1;
                std::size_t const row_zm = plane_zm + y * s1;
                std::size_t const row_zp = plane_zp + y * s1;
                std::size_t const row_wm = plane_wm + y * s1;
                std::size_t const row_wp = plane_wp + y * s1;

                // Order: next[0]+prev[0]+next[1]+prev[1]+next[2]+prev[2]+next[3]+prev[3].
                // x = 0
                {
                    std::size_t const i = row;
                    body(i,
                         data[i],
                         data[i + 1] + data[row + (L0 - 1)] + data[row_yp] + data[row_ym] +
                             data[row_zp] + data[row_zm] + data[row_wp] + data[row_wm]);
                }
                // x in (0, L0-1) — bulk
                for (std::size_t x = 1; x + 1 < L0; ++x) {
                    std::size_t const i = row + x;
                    body(i,
                         data[i],
                         data[i + 1] + data[i - 1] + data[row_yp + x] + data[row_ym + x] +
                             data[row_zp + x] + data[row_zm + x] + data[row_wp + x] +
                             data[row_wm + x]);
                }
                // x = L0-1
                {
                    std::size_t const i = row + (L0 - 1);
                    body(i,
                         data[i],
                         data[row] + data[i - 1] + data[row_yp + (L0 - 1)] +
                             data[row_ym + (L0 - 1)] + data[row_zp + (L0 - 1)] +
                             data[row_zm + (L0 - 1)] + data[row_wp + (L0 - 1)] +
                             data[row_wm + (L0 - 1)]);
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
            visit_nn_4d_(l, std::forward<Body>(body));
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

template <class T, class Acc = T, class Body>
[[nodiscard]] inline Acc reduce_fwd_4d_(Lattice<T> const& l, Body&& body) noexcept {
    RETICOLO_FP_REASSOCIATE
    T const* data        = l.data();
    auto const& sh       = l.shape();
    std::size_t const L0 = sh[0];
    std::size_t const L1 = sh[1];
    std::size_t const L2 = sh[2];
    std::size_t const L3 = sh[3];
    std::size_t const s1 = L0;
    std::size_t const s2 = L0 * L1;
    std::size_t const s3 = L0 * L1 * L2;
    Acc total{};
    for (std::size_t w = 0; w < L3; ++w) {
        std::size_t const wp     = (w + 1 == L3) ? 0 : (w + 1);
        std::size_t const hyp    = w * s3;
        std::size_t const hyp_wp = wp * s3;
        for (std::size_t z = 0; z < L2; ++z) {
            std::size_t const zp       = (z + 1 == L2) ? 0 : (z + 1);
            std::size_t const plane    = hyp + z * s2;
            std::size_t const plane_zp = hyp + zp * s2;
            std::size_t const plane_wp = hyp_wp + z * s2;
            for (std::size_t y = 0; y < L1; ++y) {
                std::size_t const yp     = (y + 1 == L1) ? 0 : (y + 1);
                std::size_t const row    = plane + y * s1;
                std::size_t const row_yp = plane + yp * s1;
                std::size_t const row_zp = plane_zp + y * s1;
                std::size_t const row_wp = plane_wp + y * s1;
                for (std::size_t x = 0; x + 1 < L0; ++x) {
                    std::size_t const i = row + x;
                    total +=
                        body(data[i],
                             data[i + 1] + data[row_yp + x] + data[row_zp + x] + data[row_wp + x]);
                }
                std::size_t const i = row + (L0 - 1);
                total += body(data[i],
                              data[row] + data[row_yp + (L0 - 1)] + data[row_zp + (L0 - 1)] +
                                  data[row_wp + (L0 - 1)]);
            }
        }
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
