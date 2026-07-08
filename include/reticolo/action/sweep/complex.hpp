#pragma once

#include <reticolo/action/sweep/stencil.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/parallel.hpp>
#include <reticolo/core/site.hpp>

#include <array>
#include <cstddef>
#include <utility>

// Split-last nearest-neighbour drivers for anisotropic complex-field actions
// (BoseGas): the last direction ("time") carries a different weight, so these
// hand the body the full 2·ndims neighbour sum AND the last-direction sum
// separately. They mirror the site engine's explicit per-dimension nests
// (<reticolo/action/sweep/stencil.hpp>) — same hoisted row bases, tiling and
// threading — but emit a second aggregate. The last direction is dim D-1, the
// OUTERMOST walk axis, so its row bases are the z/w ones already hoisted; the
// last-dim pair is one extra read per site with no inner-x wrap of its own.
//
//  visit_nn_split_last(l, body):    body(i, phi, nbrs_total, nbrs_last) -> void
//  reduce_fwd_split_last(l, body):  body(phi, fwd_total, fwd_last) -> T
//
// The isotropic combine is the identity (the anisotropy weight is applied by the
// leaf body as nbrs_total + (c_last - 1)·nbrs_last). D in {2,3,4} take the
// vectorised/threaded path; D==1 and D>4 fall back to the flat gather.

namespace reticolo::action::sweep {

// ---------- flat gather fallback (D==1, D>4) ---------------------------------

template <class T, class Body>
inline void visit_nn_split_last_fallback_(Lattice<T> const& l,
                                          std::size_t s0,
                                          std::size_t cnt,
                                          Body const& body) noexcept {
    auto const& idx              = l.indexing_ref();
    T const* data                = l.data();
    Site::value_type const* next = idx.next_data();
    Site::value_type const* prev = idx.prev_data();
    std::size_t const d          = idx.ndims();
    std::size_t const last       = d - 1;
    std::size_t const end        = s0 + cnt;

    for (std::size_t i = s0; i < end; ++i) {
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

template <class T, class Acc, class Body>
[[nodiscard]] inline Acc reduce_fwd_split_last_fallback_(Lattice<T> const& l,
                                                         std::size_t s0,
                                                         std::size_t cnt,
                                                         Body const& body) noexcept {
    RETICOLO_FP_REASSOCIATE
    auto const& idx              = l.indexing_ref();
    T const* data                = l.data();
    Site::value_type const* next = idx.next_data();
    std::size_t const d          = idx.ndims();
    std::size_t const last       = d - 1;
    std::size_t const end        = s0 + cnt;

    Acc total{};
    for (std::size_t i = s0; i < end; ++i) {
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

// ---------- per-dimension split-last item nests -------------------------------
//
// Same explicit hoisted nests as the site engine, but each site emits TWO sums:
// the full 2·ndims neighbour total AND the last-direction (dim D-1) pair on its
// own. The last direction is the outermost walk axis, so its row bases are the
// z/w ones already hoisted. Identity combine (raw reads). Total order and the +x
// peel match the gather fallback → bit-identical for any thread count.

template <class T, class Body>
inline void map_split_2d_(T const* data,
                               std::array<std::size_t, 2> const& L,
                               std::array<std::size_t, 2> const& stride,
                               std::array<std::size_t, 2> const& lo,
                               std::array<std::size_t, 2> const& hi,
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
        auto const emit = [&](std::size_t i, std::size_t xm, std::size_t xp, std::size_t off) {
            T const self = data[i];
            T const last = data[row_yp + off] + data[row_ym + off];
            body(i, self, data[xp] + data[xm] + last, last);
        };
        emit(row, row + (L0 - 1), row + 1, 0);
        for (std::size_t x = 1; x + 1 < L0; ++x) {
            emit(row + x, row + x - 1, row + x + 1, x);
        }
        emit(row + (L0 - 1), row + (L0 - 2), row, L0 - 1);
    }
}

template <class Acc, class T, class Body>
[[nodiscard]] inline Acc reduce_split_2d_(T const* data,
                                               std::array<std::size_t, 2> const& L,
                                               std::array<std::size_t, 2> const& stride,
                                               std::array<std::size_t, 2> const& lo,
                                               std::array<std::size_t, 2> const& hi,
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
            T const last        = data[row_yp + x];
            total += body(data[i], data[i + 1] + last, last);
        }
        std::size_t const i = row + (L0 - 1);
        T const last        = data[row_yp + (L0 - 1)];
        total += body(data[i], data[row] + last, last);
    }
    return total;
}

template <class T, class Body>
inline void map_split_3d_(T const* data,
                               std::array<std::size_t, 3> const& L,
                               std::array<std::size_t, 3> const& stride,
                               std::array<std::size_t, 3> const& lo,
                               std::array<std::size_t, 3> const& hi,
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
            auto const emit = [&](std::size_t i, std::size_t xm, std::size_t xp, std::size_t off) {
                T const self = data[i];
                T const last = data[row_zp + off] + data[row_zm + off];
                body(i, self,
                     data[xp] + data[xm] + data[row_yp + off] + data[row_ym + off] + last, last);
            };
            emit(row, row + (L0 - 1), row + 1, 0);
            for (std::size_t x = 1; x + 1 < L0; ++x) {
                emit(row + x, row + x - 1, row + x + 1, x);
            }
            emit(row + (L0 - 1), row + (L0 - 2), row, L0 - 1);
        }
    }
}

template <class Acc, class T, class Body>
[[nodiscard]] inline Acc reduce_split_3d_(T const* data,
                                               std::array<std::size_t, 3> const& L,
                                               std::array<std::size_t, 3> const& stride,
                                               std::array<std::size_t, 3> const& lo,
                                               std::array<std::size_t, 3> const& hi,
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
                T const last        = data[row_zp + x];
                total += body(data[i], data[i + 1] + data[row_yp + x] + last, last);
            }
            std::size_t const i = row + (L0 - 1);
            T const last        = data[row_zp + (L0 - 1)];
            total += body(data[i], data[row] + data[row_yp + (L0 - 1)] + last, last);
        }
    }
    return total;
}

template <class T, class Body>
inline void map_split_4d_(T const* data,
                               std::array<std::size_t, 4> const& L,
                               std::array<std::size_t, 4> const& stride,
                               std::array<std::size_t, 4> const& lo,
                               std::array<std::size_t, 4> const& hi,
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
                auto const emit = [&](std::size_t i, std::size_t xm, std::size_t xp,
                                      std::size_t off) {
                    T const self = data[i];
                    T const last = data[row_wp + off] + data[row_wm + off];
                    body(i, self,
                         data[xp] + data[xm] + data[row_yp + off] + data[row_ym + off] +
                             data[row_zp + off] + data[row_zm + off] + last,
                         last);
                };
                emit(row, row + (L0 - 1), row + 1, 0);
                for (std::size_t x = 1; x + 1 < L0; ++x) {
                    emit(row + x, row + x - 1, row + x + 1, x);
                }
                emit(row + (L0 - 1), row + (L0 - 2), row, L0 - 1);
            }
        }
    }
}

template <class Acc, class T, class Body>
[[nodiscard]] inline Acc reduce_split_4d_(T const* data,
                                               std::array<std::size_t, 4> const& L,
                                               std::array<std::size_t, 4> const& stride,
                                               std::array<std::size_t, 4> const& lo,
                                               std::array<std::size_t, 4> const& hi,
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
                    T const last        = data[row_wp + x];
                    total += body(data[i],
                                  data[i + 1] + data[row_yp + x] + data[row_zp + x] + last, last);
                }
                std::size_t const i = row + (L0 - 1);
                T const last        = data[row_wp + (L0 - 1)];
                total += body(data[i],
                              data[row] + data[row_yp + (L0 - 1)] + data[row_zp + (L0 - 1)] + last,
                              last);
            }
        }
    }
    return total;
}

// ---------- dispatch ---------------------------------------------------------

//  visit_nn_split_last(l, body): body(i, phi, nbrs_total, nbrs_last) -> void.
//  Reuses the shared `traverse_dispatch_` (site and split-last share the tiling and
//  partition); the split item drivers add only the second (last-direction)
//  aggregate. 1D and D>4 route to the flat split-last gather.
template <class T, class Body>
inline void visit_nn_split_last(Lattice<T> const& l, Body&& body) noexcept {
    Body const& b = body;
    auto flat     = [&](std::size_t s0, std::size_t cnt) {
        visit_nn_split_last_fallback_(l, s0, cnt, b);
    };
    traverse_dispatch_<void>(
        l,
        [&]<std::size_t D>(auto const& L, auto const& stride, auto const& lo, auto const& hi) {
            if constexpr (D == 2) {
                map_split_2d_(l.data(), L, stride, lo, hi, b);
            } else if constexpr (D == 3) {
                map_split_3d_(l.data(), L, stride, lo, hi, b);
            } else {
                map_split_4d_(l.data(), L, stride, lo, hi, b);
            }
        },
        flat,
        flat);
}

//  reduce_fwd_split_last(l, body): body(phi, fwd_total, fwd_last) -> T.
template <class T, class Body>
[[nodiscard]] inline T reduce_fwd_split_last(Lattice<T> const& l, Body&& body) noexcept {
    Body const& b = body;
    auto flat     = [&](std::size_t s0, std::size_t cnt) {
        return reduce_fwd_split_last_fallback_<T, T>(l, s0, cnt, b);
    };
    return traverse_dispatch_<T>(
        l,
        [&]<std::size_t D>(auto const& L, auto const& stride, auto const& lo, auto const& hi) {
            if constexpr (D == 2) {
                return reduce_split_2d_<T>(l.data(), L, stride, lo, hi, b);
            } else if constexpr (D == 3) {
                return reduce_split_3d_<T>(l.data(), L, stride, lo, hi, b);
            } else {
                return reduce_split_4d_<T>(l.data(), L, stride, lo, hi, b);
            }
        },
        flat,
        flat);
}

}  // namespace reticolo::action::sweep
