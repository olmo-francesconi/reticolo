#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/site.hpp>

#include <cstddef>
#include <utility>

// Split-last nearest-neighbour drivers for anisotropic complex-field actions
// (BoseGas): the last direction ("time") carries a different weight, so these
// hand the body the full 2*ndims neighbour sum AND the last-direction sum
// separately. Twin of the isotropic site drivers in ../../site/detail/traversal.hpp.

#ifndef RETICOLO_FP_REASSOCIATE
    #if defined(__clang__)
        #define RETICOLO_FP_REASSOCIATE _Pragma("clang fp reassociate(on)")
    #else
        #define RETICOLO_FP_REASSOCIATE
    #endif
#endif

namespace reticolo::action::detail {

// ---------- visit_nn_split_last ---------------------------------------------
//
//  visit_nn_split_last(l, body): body(i, phi, nbrs_total, nbrs_last) -> void.
//  nbrs_total is the unweighted sum of all 2*ndims NN (same as visit_nn).
//  nbrs_last  is the sum of the 2 NN along the LAST direction only.
//
//  Used by actions with an anisotropy on one direction (typically the time
//  direction at finite chemical potential / temperature). The body composes
//  the weighted staple as nbrs_total + (c_last - 1) * nbrs_last so the helper
//  does not need to know the physics weights.

template <class T, class Body>
inline void visit_nn_split_last_fallback_(Lattice<T> const& l, Body&& body) noexcept {
    auto const& idx              = l.indexing_ref();
    T const* data                = l.data();
    Site::value_type const* next = idx.next_data();
    Site::value_type const* prev = idx.prev_data();
    std::size_t const n          = idx.nsites();
    std::size_t const d          = idx.ndims();
    std::size_t const last       = d - 1;

    for (std::size_t i = 0; i < n; ++i) {
        T nbrs                 = T{0};
        std::size_t const base = i * d;
        for (std::size_t mu = 0; mu < d; ++mu) {
            nbrs += data[next[base + mu]];
            nbrs += data[prev[base + mu]];
        }
        T const nbrs_last = data[next[base + last]] + data[prev[base + last]];
        body(i, data[i], nbrs, nbrs_last);
    }
}

template <class T, class Body>
inline void visit_nn_split_last_3d_(Lattice<T> const& l, Body&& body) noexcept {
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

            // x = 0
            {
                std::size_t const i = row;
                T const last_pair   = data[row_zp] + data[row_zm];
                body(i,
                     data[i],
                     data[i + 1] + data[row + (Lx - 1)] + data[row_yp] + data[row_ym] + last_pair,
                     last_pair);
            }
            // x in (0, Lx-1) — bulk
            for (std::size_t x = 1; x + 1 < Lx; ++x) {
                std::size_t const i = row + x;
                T const last_pair   = data[row_zp + x] + data[row_zm + x];
                body(i,
                     data[i],
                     data[i + 1] + data[i - 1] + data[row_yp + x] + data[row_ym + x] + last_pair,
                     last_pair);
            }
            // x = Lx-1
            {
                std::size_t const i = row + (Lx - 1);
                T const last_pair   = data[row_zp + (Lx - 1)] + data[row_zm + (Lx - 1)];
                body(i,
                     data[i],
                     data[row] + data[i - 1] + data[row_yp + (Lx - 1)] + data[row_ym + (Lx - 1)] +
                         last_pair,
                     last_pair);
            }
        }
    }
}

template <class T, class Body>
inline void visit_nn_split_last_4d_(Lattice<T> const& l, Body&& body) noexcept {
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

                // x = 0
                {
                    std::size_t const i = row;
                    T const last_pair   = data[row_wp] + data[row_wm];
                    body(i,
                         data[i],
                         data[i + 1] + data[row + (L0 - 1)] + data[row_yp] + data[row_ym] +
                             data[row_zp] + data[row_zm] + last_pair,
                         last_pair);
                }
                // x in (0, L0-1) — bulk
                for (std::size_t x = 1; x + 1 < L0; ++x) {
                    std::size_t const i = row + x;
                    T const last_pair   = data[row_wp + x] + data[row_wm + x];
                    body(i,
                         data[i],
                         data[i + 1] + data[i - 1] + data[row_yp + x] + data[row_ym + x] +
                             data[row_zp + x] + data[row_zm + x] + last_pair,
                         last_pair);
                }
                // x = L0-1
                {
                    std::size_t const i = row + (L0 - 1);
                    T const last_pair   = data[row_wp + (L0 - 1)] + data[row_wm + (L0 - 1)];
                    body(i,
                         data[i],
                         data[row] + data[i - 1] + data[row_yp + (L0 - 1)] +
                             data[row_ym + (L0 - 1)] + data[row_zp + (L0 - 1)] +
                             data[row_zm + (L0 - 1)] + last_pair,
                         last_pair);
                }
            }
        }
    }
}

template <class T, class Body>
inline void visit_nn_split_last(Lattice<T> const& l, Body&& body) noexcept {
#if RETICOLO_HOT_LOOP_FORCE_FALLBACK
    visit_nn_split_last_fallback_(l, std::forward<Body>(body));
#else
    switch (l.ndims()) {
        case 3:
            visit_nn_split_last_3d_(l, std::forward<Body>(body));
            return;
        case 4:
            visit_nn_split_last_4d_(l, std::forward<Body>(body));
            return;
        default:
            visit_nn_split_last_fallback_(l, std::forward<Body>(body));
            return;
    }
#endif
}

// ---------- reduce_fwd_split_last -------------------------------------------
//
//  reduce_fwd_split_last(l, body): body(phi, fwd_total, fwd_last) -> T.
//  Forward-only twin of visit_nn_split_last for s_full of anisotropic actions.

template <class T, class Body>
[[nodiscard]] inline T reduce_fwd_split_last_fallback_(Lattice<T> const& l, Body&& body) noexcept {
    RETICOLO_FP_REASSOCIATE
    auto const& idx              = l.indexing_ref();
    T const* data                = l.data();
    Site::value_type const* next = idx.next_data();
    std::size_t const n          = idx.nsites();
    std::size_t const d          = idx.ndims();
    std::size_t const last       = d - 1;

    T total{};
    for (std::size_t i = 0; i < n; ++i) {
        T fwd_sum              = T{0};
        std::size_t const base = i * d;
        for (std::size_t mu = 0; mu < d; ++mu) {
            fwd_sum += data[next[base + mu]];
        }
        T const fwd_last = data[next[base + last]];
        total += body(data[i], fwd_sum, fwd_last);
    }
    return total;
}

template <class T, class Body>
[[nodiscard]] inline T reduce_fwd_split_last_3d_(Lattice<T> const& l, Body&& body) noexcept {
    RETICOLO_FP_REASSOCIATE
    T const* data        = l.data();
    auto const& sh       = l.shape();
    std::size_t const Lx = sh[0];
    std::size_t const Ly = sh[1];
    std::size_t const Lz = sh[2];
    std::size_t const sy = Lx;
    std::size_t const sz = Lx * Ly;
    T total{};
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
                T const fwd_last    = data[row_zp + x];
                total += body(data[i], data[i + 1] + data[row_yp + x] + fwd_last, fwd_last);
            }
            std::size_t const i   = row + (Lx - 1);
            T const fwd_last_last = data[row_zp + (Lx - 1)];
            total +=
                body(data[i], data[row] + data[row_yp + (Lx - 1)] + fwd_last_last, fwd_last_last);
        }
    }
    return total;
}

template <class T, class Body>
[[nodiscard]] inline T reduce_fwd_split_last_4d_(Lattice<T> const& l, Body&& body) noexcept {
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
    T total{};
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
                    T const fwd_last    = data[row_wp + x];
                    total += body(data[i],
                                  data[i + 1] + data[row_yp + x] + data[row_zp + x] + fwd_last,
                                  fwd_last);
                }
                std::size_t const i   = row + (L0 - 1);
                T const fwd_last_last = data[row_wp + (L0 - 1)];
                total += body(data[i],
                              data[row] + data[row_yp + (L0 - 1)] + data[row_zp + (L0 - 1)] +
                                  fwd_last_last,
                              fwd_last_last);
            }
        }
    }
    return total;
}

template <class T, class Body>
[[nodiscard]] inline T reduce_fwd_split_last(Lattice<T> const& l, Body&& body) noexcept {
#if RETICOLO_HOT_LOOP_FORCE_FALLBACK
    return reduce_fwd_split_last_fallback_(l, std::forward<Body>(body));
#else
    switch (l.ndims()) {
        case 3:
            return reduce_fwd_split_last_3d_(l, std::forward<Body>(body));
        case 4:
            return reduce_fwd_split_last_4d_(l, std::forward<Body>(body));
        default:
            return reduce_fwd_split_last_fallback_(l, std::forward<Body>(body));
    }
#endif
}

}  // namespace reticolo::action::detail
