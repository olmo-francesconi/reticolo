#pragma once

#include <reticolo/action/detail/traverse.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/parallel.hpp>
#include <reticolo/core/site.hpp>

#include <array>
#include <cstddef>
#include <utility>

// Split-last nearest-neighbour drivers for anisotropic complex-field actions
// (BoseGas): the last direction ("time") carries a different weight, so these
// hand the body the full 2·ndims neighbour sum AND the last-direction sum
// separately. They reuse the shared dimension-generic engine
// (<reticolo/action/detail/traverse.hpp>) — its tiling, per-dim vectorised
// neighbour-base geometry (`walk_outer_`, `agg_*`), and threading — adding only
// the second (last-direction) aggregate. The last direction is dim D-1, which is
// the OUTERMOST axis of the walk, so its pre-wrapped row bases fwd[D-1]/bwd[D-1]
// are already produced by the engine; the last-dim pair is one extra read per
// site with no inner-x wrap of its own.
//
//  visit_nn_split_last(l, body):    body(i, phi, nbrs_total, nbrs_last) -> void
//  reduce_fwd_split_last(l, body):  body(phi, fwd_total, fwd_last) -> T
//
// The isotropic combine is the identity (the anisotropy weight is applied by the
// leaf body as nbrs_total + (c_last - 1)·nbrs_last). D in {2,3,4} take the
// vectorised/threaded path; D==1 and D>4 fall back to the flat gather.

namespace reticolo::action::detail {

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

// ---------- split-last row + item (reuse shared agg_*/walk_outer_) ------------

// One innermost row [0, L0): the TOTAL reuses the shared identity-combine
// aggregate (inner-x wrap peeled); the last-direction pair is a plain read at the
// dim-(D-1) row bases (uniform in x, no inner wrap of its own).
template <std::size_t D, class T, class Body>
inline void map_row_split_(T const* data,
                           std::size_t own,
                           std::array<std::size_t, D> const& fwd,
                           std::array<std::size_t, D> const& bwd,
                           std::size_t L0,
                           Body const& body) noexcept {
    constexpr std::size_t last = D - 1;
    IdentityCombine const id{};
    {
        T const self  = data[own];
        T const total = agg_lo_<D, AllDirs>(data, self, own, L0, fwd, bwd, id);
        body(own, self, total, data[fwd[last]] + data[bwd[last]]);
    }
    for (std::size_t x = 1; x + 1 < L0; ++x) {
        std::size_t const i = own + x;
        T const self        = data[i];
        T const total       = agg_bulk_<D, AllDirs>(data, self, i, x, fwd, bwd, id);
        body(i, self, total, data[fwd[last] + x] + data[bwd[last] + x]);
    }
    {
        std::size_t const i = own + (L0 - 1);
        T const self        = data[i];
        T const total       = agg_hi_<D, AllDirs>(data, self, own, L0, fwd, bwd, id);
        body(i, self, total, data[fwd[last] + (L0 - 1)] + data[bwd[last] + (L0 - 1)]);
    }
}

template <std::size_t D, class Acc, class T, class Body>
[[nodiscard]] inline Acc reduce_row_split_(T const* data,
                                           std::size_t own,
                                           std::array<std::size_t, D> const& fwd,
                                           std::array<std::size_t, D> const& bwd,
                                           std::size_t L0,
                                           Body const& body) noexcept {
    RETICOLO_FP_REASSOCIATE
    constexpr std::size_t last = D - 1;
    IdentityCombine const id{};
    Acc total{};
    for (std::size_t x = 0; x + 1 < L0; ++x) {
        std::size_t const i = own + x;
        T const self        = data[i];
        T const fwd_total   = agg_bulk_<D, FwdOnly>(data, self, i, x, fwd, bwd, id);
        total += body(self, fwd_total, data[fwd[last] + x]);
    }
    {
        std::size_t const i = own + (L0 - 1);
        T const self        = data[i];
        T const fwd_total   = agg_hi_<D, FwdOnly>(data, self, own, L0, fwd, bwd, id);
        total += body(self, fwd_total, data[fwd[last] + (L0 - 1)]);
    }
    return total;
}

template <std::size_t D, class T, class Body>
inline void map_item_split_(Lattice<T> const& l,
                            std::array<std::size_t, D> const& L,
                            std::array<std::size_t, D> const& stride,
                            std::array<std::size_t, D> const& lo,
                            std::array<std::size_t, D> const& hi,
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
            map_row_split_<D>(data, own, fwd, bwd, L0, body);
        });
}

template <std::size_t D, class Acc, class T, class Body>
[[nodiscard]] inline Acc reduce_item_split_(Lattice<T> const& l,
                                            std::array<std::size_t, D> const& L,
                                            std::array<std::size_t, D> const& stride,
                                            std::array<std::size_t, D> const& lo,
                                            std::array<std::size_t, D> const& hi,
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
            total += reduce_row_split_<D, Acc>(data, own, fwd, bwd, L0, body);
        });
    return total;
}

// ---------- dispatch ---------------------------------------------------------

//  visit_nn_split_last(l, body): body(i, phi, nbrs_total, nbrs_last) -> void.
template <class T, class Body>
inline void visit_nn_split_last(Lattice<T> const& l, Body&& body) noexcept {
    std::size_t const n   = l.nsites();
    std::size_t const bps = l.bytes_per_site();
    bool const want       = reticolo::detail::want_threads(n, bps);
    Body const& b         = body;
#if RETICOLO_HOT_LOOP_FORCE_FALLBACK
    reticolo::detail::parallel_map_ranges(n, bps, 1, [&](std::size_t s0, std::size_t cnt) {
        visit_nn_split_last_fallback_(l, s0, cnt, b);
    });
#else
    switch (l.ndims()) {
        case 2: {
            auto const [L, stride] = geometry_<2>(l);
            reticolo::detail::parallel_map(want, L[1], [&](std::size_t y) {
                std::array<std::size_t, 2> const lo{0, y};
                std::array<std::size_t, 2> const hi{L[0], y + 1};
                map_item_split_<2>(l, L, stride, lo, hi, b);
            });
            return;
        }
        case 3: {
            auto const [L, stride] = geometry_<3>(l);
            reticolo::detail::parallel_map(want, L[2], [&](std::size_t z) {
                std::array<std::size_t, 3> const lo{0, 0, z};
                std::array<std::size_t, 3> const hi{L[0], L[1], z + 1};
                map_item_split_<3>(l, L, stride, lo, hi, b);
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
                map_item_split_<4>(l, L, stride, lo, hi, b);
            });
            return;
        }
        default:
            reticolo::detail::parallel_map_ranges(n, bps, 1, [&](std::size_t s0, std::size_t cnt) {
                visit_nn_split_last_fallback_(l, s0, cnt, b);
            });
            return;
    }
#endif
}

//  reduce_fwd_split_last(l, body): body(phi, fwd_total, fwd_last) -> T.
template <class T, class Body>
[[nodiscard]] inline T reduce_fwd_split_last(Lattice<T> const& l, Body&& body) noexcept {
    std::size_t const n   = l.nsites();
    std::size_t const bps = l.bytes_per_site();
    bool const want       = reticolo::detail::want_threads(n, bps);
    Body const& b         = body;
#if RETICOLO_HOT_LOOP_FORCE_FALLBACK
    return reticolo::detail::parallel_reduce_ranges<T>(
        n, bps, 1, [&](std::size_t s0, std::size_t cnt) {
            return reduce_fwd_split_last_fallback_<T, T>(l, s0, cnt, b);
        });
#else
    switch (l.ndims()) {
        case 2: {
            auto const [L, stride] = geometry_<2>(l);
            return reticolo::detail::parallel_reduce<T>(want, L[1], [&](std::size_t y) {
                std::array<std::size_t, 2> const lo{0, y};
                std::array<std::size_t, 2> const hi{L[0], y + 1};
                return reduce_item_split_<2, T>(l, L, stride, lo, hi, b);
            });
        }
        case 3: {
            auto const [L, stride] = geometry_<3>(l);
            return reticolo::detail::parallel_reduce<T>(want, L[2], [&](std::size_t z) {
                std::array<std::size_t, 3> const lo{0, 0, z};
                std::array<std::size_t, 3> const hi{L[0], L[1], z + 1};
                return reduce_item_split_<3, T>(l, L, stride, lo, hi, b);
            });
        }
        case 4: {
            auto const [L, stride] = geometry_<4>(l);
            Block4d const bl       = plan_block_4d_<T>(L[0], L[1], L[2], L[3]);
            std::size_t const nty  = (L[1] + bl.by - 1) / bl.by;
            std::size_t const ntz  = (L[2] + bl.bz - 1) / bl.bz;
            std::size_t const ntw  = (L[3] + bl.bw - 1) / bl.bw;
            return reticolo::detail::parallel_reduce<T>(
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
                    return reduce_item_split_<4, T>(l, L, stride, lo, hi, b);
                });
        }
        default:
            return reticolo::detail::parallel_reduce_ranges<T>(
                n, bps, 1, [&](std::size_t s0, std::size_t cnt) {
                    return reduce_fwd_split_last_fallback_<T, T>(l, s0, cnt, b);
                });
    }
#endif
}

}  // namespace reticolo::action::detail
