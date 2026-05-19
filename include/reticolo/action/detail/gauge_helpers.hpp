#pragma once

#include <reticolo/core/indexing.hpp>
#include <reticolo/core/link_lattice.hpp>

#include <cstddef>
#include <utility>

namespace reticolo::gauge::detail {

// =============================================================================
//  Plaquette-plane bulk-vs-slab hot-loop helper for direction-major
//  LinkLattice<T>. Each call iterates one (mu, nu) plane and invokes the body
//  on every site s with the precomputed neighbour indices
//      body(s, s_pmu, s_pnu)
//  where  s_pmu = next(s, mu),  s_pnu = next(s, nu)  (both periodic).
//
//  In the bulk — sites where neither s + mu_hat nor s + nu_hat wraps — the
//  offsets `s_pmu - s` and `s_pnu - s` are *plane-constants* (the strides of
//  mu and nu). The inner loop becomes stride-1 in s with fixed offsets, so
//  reads of `mu_block[s + off_*]` and writes scattered to `mom_*_block[
//  s + off_*]` autovectorise. The wrap boundaries (a thin slab in each of
//  mu, nu and the joint corner) are peeled off and handled with `next[]`.
//
//  Specialisations: 2D, 3D, 4D. Fallback uses `next[]` for every access.
// =============================================================================

template <class T, class Body>
inline void visit_plane_fallback_(LinkLattice<T> const& l,
                                  std::size_t mu,
                                  std::size_t nu,
                                  Body const& body) noexcept {
    Indexing const& idx                = l.indexing_ref();
    Site::value_type const* next_table = idx.next_data();
    std::size_t const ns               = l.nsites();
    std::size_t const d                = idx.ndims();
    for (std::size_t s = 0; s < ns; ++s) {
        std::size_t const s_pmu = next_table[(s * d) + mu];
        std::size_t const s_pnu = next_table[(s * d) + nu];
        body(s, s_pmu, s_pnu);
    }
}

// 3D plane (mu, nu): factor the lattice as (x, y, z) and iterate outer-outer
// over the two coordinates that are *not* coord[0]. Inner loop runs coord[0]
// stride-1. If 0 in {mu, nu} the inner loop has a wrap slab at x = L0 - 1;
// otherwise the inner loop is wrap-free (all bulk).
template <class T, class Body>
inline void visit_plane_3d_(LinkLattice<T> const& l,
                            std::size_t mu,
                            std::size_t nu,
                            Body const& body) noexcept {
    auto const& sh       = l.shape();
    std::size_t const L0 = sh[0];
    std::size_t const L1 = sh[1];
    std::size_t const L2 = sh[2];
    std::size_t const sy = L0;
    std::size_t const sz = L0 * L1;

    auto wrap_lo = [&](std::size_t dir, std::size_t c) -> std::size_t {
        // returns -wrap-offset (positive: subtract this from base when wrapping)
        if (dir == 0)
            return c * 1;
        if (dir == 1)
            return c * sy;
        return c * sz;
    };

    // Plane (1, 2): inner x has no wrap.
    if (mu == 1 && nu == 2) {
        for (std::size_t z = 0; z < L2; ++z) {
            std::size_t const zp = (z + 1 == L2) ? 0 : (z + 1);
            for (std::size_t y = 0; y < L1; ++y) {
                std::size_t const yp     = (y + 1 == L1) ? 0 : (y + 1);
                std::size_t const row    = z * sz + y * sy;
                std::size_t const row_yp = z * sz + yp * sy;
                std::size_t const row_zp = zp * sz + y * sy;
                for (std::size_t x = 0; x < L0; ++x) {
                    std::size_t const s     = row + x;
                    std::size_t const s_pmu = row_yp + x;
                    std::size_t const s_pnu = row_zp + x;
                    body(s, s_pmu, s_pnu);
                }
            }
        }
        return;
    }

    // Plane (0, 1) or (0, 2): inner x has wrap at L0-1.
    if (mu == 0 && (nu == 1 || nu == 2)) {
        std::size_t const stride_nu = (nu == 1) ? sy : sz;
        std::size_t const len_nu    = (nu == 1) ? L1 : L2;
        for (std::size_t z = 0; z < L2; ++z) {
            for (std::size_t y = 0; y < L1; ++y) {
                std::size_t const row = z * sz + y * sy;
                std::size_t cnu;
                if (nu == 1)
                    cnu = y;
                else
                    cnu = z;
                std::size_t const cnu_p = (cnu + 1 < len_nu) ? cnu + 1 : 0;
                std::size_t row_pnu;
                if (nu == 1)
                    row_pnu = z * sz + cnu_p * sy;
                else
                    row_pnu = cnu_p * sz + y * sy;

                // bulk x in [0, L0-1)
                for (std::size_t x = 0; x + 1 < L0; ++x) {
                    std::size_t const s     = row + x;
                    std::size_t const s_pmu = s + 1;
                    std::size_t const s_pnu = row_pnu + x;
                    body(s, s_pmu, s_pnu);
                }
                // x = L0 - 1 (wrap in mu = 0)
                {
                    std::size_t const s     = row + (L0 - 1);
                    std::size_t const s_pmu = row;
                    std::size_t const s_pnu = row_pnu + (L0 - 1);
                    body(s, s_pmu, s_pnu);
                }
            }
        }
        (void)stride_nu;
        (void)wrap_lo;
        return;
    }

    visit_plane_fallback_(l, mu, nu, body);
}

template <class T, class Body>
inline void visit_plane_4d_(LinkLattice<T> const& l,
                            std::size_t mu,
                            std::size_t nu,
                            Body const& body) noexcept {
    auto const& sh       = l.shape();
    std::size_t const L0 = sh[0];
    std::size_t const L1 = sh[1];
    std::size_t const L2 = sh[2];
    std::size_t const L3 = sh[3];
    std::size_t const sy = L0;
    std::size_t const sz = L0 * L1;
    std::size_t const sw = L0 * L1 * L2;

    auto row_at = [&](std::size_t y, std::size_t z, std::size_t w) -> std::size_t {
        return w * sw + z * sz + y * sy;
    };

    // mu == 0 family: inner-x has a wrap slab at L0-1.
    if (mu == 0 && (nu == 1 || nu == 2 || nu == 3)) {
        for (std::size_t w = 0; w < L3; ++w) {
            for (std::size_t z = 0; z < L2; ++z) {
                for (std::size_t y = 0; y < L1; ++y) {
                    std::size_t const row = row_at(y, z, w);
                    std::size_t row_pnu;
                    switch (nu) {
                        case 1: {
                            std::size_t const yp = (y + 1 == L1) ? 0 : (y + 1);
                            row_pnu              = row_at(yp, z, w);
                            break;
                        }
                        case 2: {
                            std::size_t const zp = (z + 1 == L2) ? 0 : (z + 1);
                            row_pnu              = row_at(y, zp, w);
                            break;
                        }
                        default: {
                            std::size_t const wp = (w + 1 == L3) ? 0 : (w + 1);
                            row_pnu              = row_at(y, z, wp);
                            break;
                        }
                    }
                    for (std::size_t x = 0; x + 1 < L0; ++x) {
                        std::size_t const s     = row + x;
                        std::size_t const s_pmu = s + 1;
                        std::size_t const s_pnu = row_pnu + x;
                        body(s, s_pmu, s_pnu);
                    }
                    {
                        std::size_t const s     = row + (L0 - 1);
                        std::size_t const s_pmu = row;
                        std::size_t const s_pnu = row_pnu + (L0 - 1);
                        body(s, s_pmu, s_pnu);
                    }
                }
            }
        }
        return;
    }

    // mu, nu both > 0: inner-x is wrap-free.
    if (mu >= 1 && nu > mu && nu <= 3) {
        for (std::size_t w = 0; w < L3; ++w) {
            for (std::size_t z = 0; z < L2; ++z) {
                for (std::size_t y = 0; y < L1; ++y) {
                    std::size_t const row = row_at(y, z, w);

                    std::size_t row_pmu;
                    switch (mu) {
                        case 1: {
                            std::size_t const yp = (y + 1 == L1) ? 0 : (y + 1);
                            row_pmu              = row_at(yp, z, w);
                            break;
                        }
                        case 2: {
                            std::size_t const zp = (z + 1 == L2) ? 0 : (z + 1);
                            row_pmu              = row_at(y, zp, w);
                            break;
                        }
                        default: {
                            std::size_t const wp = (w + 1 == L3) ? 0 : (w + 1);
                            row_pmu              = row_at(y, z, wp);
                            break;
                        }
                    }
                    std::size_t row_pnu;
                    switch (nu) {
                        case 2: {
                            std::size_t const zp = (z + 1 == L2) ? 0 : (z + 1);
                            row_pnu              = row_at(y, zp, w);
                            break;
                        }
                        default: {  // nu == 3
                            std::size_t const wp = (w + 1 == L3) ? 0 : (w + 1);
                            row_pnu              = row_at(y, z, wp);
                            break;
                        }
                    }
                    for (std::size_t x = 0; x < L0; ++x) {
                        std::size_t const s     = row + x;
                        std::size_t const s_pmu = row_pmu + x;
                        std::size_t const s_pnu = row_pnu + x;
                        body(s, s_pmu, s_pnu);
                    }
                }
            }
        }
        return;
    }

    visit_plane_fallback_(l, mu, nu, body);
}

template <class T, class Body>
inline void
visit_plane(LinkLattice<T> const& l, std::size_t mu, std::size_t nu, Body const& body) noexcept {
    switch (l.ndims()) {
        case 3:
            visit_plane_3d_(l, mu, nu, body);
            return;
        case 4:
            visit_plane_4d_(l, mu, nu, body);
            return;
        default:
            visit_plane_fallback_(l, mu, nu, body);
            return;
    }
}

}  // namespace reticolo::gauge::detail
