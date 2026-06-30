#pragma once

#include <reticolo/action/detail/gauge/compact_u1_formula.hpp>
#include <reticolo/action/detail/gauge/gauge_helpers.hpp>
#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/indexing.hpp>
#include <reticolo/core/link_lattice.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/math/vec_libm.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace reticolo::action {

// Compact U(1) gauge theory with the standard Wilson plaquette action:
//
//     S_W = beta * sum_x  sum_{mu<nu}  ( 1 - cos( theta_{mu,nu}(x) ) )
//
//     theta_{mu,nu}(x) =  theta_mu(x)
//                       + theta_nu(x + mu_hat)
//                       - theta_mu(x + nu_hat)
//                       - theta_nu(x)
//
// This is the convention used by every production lattice code (openQCD,
// Grid, MILC, Chroma, QUDA) and by Gattringer-Lang. S_W is bounded below
// by 0 (cold config), reaches `beta * n_plaq` at maximal disorder.
//
// Boltzmann weight: exp(-S_W). The gauge HMC uses H = K + S so
// exp(-H) = exp(-K - S). `compute_force` returns the *force* F = -dS/dtheta;
// the integrator kick is `mom += k_dt * F`.
//
// The (1 - cos) constant `beta * n_plaq_total` on `s_full` makes it
// non-negative for sanity checking — the HMC dynamics only depend on
// differences, where the constant cancels.
//
// Storage layout: `LinkLattice<T>` is direction-major (each direction is a
// contiguous nsites-element block). The hot loops here iterate one
// plaquette plane (mu, nu) at a time via `gauge::detail::visit_plane`,
// which peels wrap slabs off the boundary so the bulk body sees stride-1
// reads and writes through fixed `(s_pmu - s, s_pnu - s)` offsets — clean
// autovectorisation territory.

template <class T = double>
struct CompactU1 {
    using value_type = T;
    using field_type = LinkLattice<T>;

    T beta = T{0};

    void describe(log::Entry& e) const {
        e.line("CompactU1<{}>", scalar_name<T>());
        e.param("β={:.3f}", beta);
    }

    // Per-plaquette cos in `T`, plaquette sum accumulated in (and returned as)
    // `double` — see Phi4::s_full for the rationale.
    [[nodiscard]] double s_full(LinkLattice<T> const& l) const noexcept {
        std::size_t const d      = l.ndims();
        std::size_t const n      = l.nsites();
        std::size_t const n_plaq = (d * (d - 1) / 2) * n;
        double accum             = 0.0;
        ensure_scratch_(n);
        T* const buf = scratch.data();
        for (std::size_t mu = 0; mu < d; ++mu) {
            T const* mb = l.mu_data(mu);
            for (std::size_t nu = mu + 1; nu < d; ++nu) {
                T const* nb = l.mu_data(nu);
                // Pass 1: stash plaquette angles for the plane into the scratch.
                detail::visit_plane(
                    l, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                        buf[s] = detail::u1_plaq(mb[s], nb[s_pmu], mb[s_pnu], nb[s]);
                    });
                // Vector cos in place.
                math::cos_batch(buf, buf, n);
                // Tree-reducible scalar reduction; reassociate so the
                // OoO engine can fold multiple lanes in parallel.
                {
#if defined(__clang__)
                    _Pragma("clang fp reassociate(on)")
#endif
                        for (std::size_t s = 0; s < n; ++s) {
                        accum += static_cast<double>(buf[s]);
                    }
                }
            }
        }
        // Standard Wilson form: S = beta * sum (1 - cos theta_p).
        double const s = static_cast<double>(beta) * (static_cast<double>(n_plaq) - accum);
        last_s_full_   = s;
        return s;
    }

    [[nodiscard]] double last_s_full() const noexcept { return last_s_full_; }
    void restore_last_s_full(double v) const noexcept { last_s_full_ = v; }

    void compute_force(LinkLattice<T> const& l, LinkLattice<T>& force) const noexcept {
        std::fill(force.begin(), force.end(), T{0});
        std::size_t const d = l.ndims();
        std::size_t const n = l.nsites();
        T const b           = beta;
        ensure_scratch_(n);
        T* const buf = scratch.data();
        for (std::size_t mu = 0; mu < d; ++mu) {
            T const* mb  = l.mu_data(mu);
            T* const fmu = force.mu_data(mu);
            for (std::size_t nu = mu + 1; nu < d; ++nu) {
                T const* nb  = l.mu_data(nu);
                T* const fnu = force.mu_data(nu);
                detail::visit_plane(
                    l, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                        buf[s] = detail::u1_plaq(mb[s], nb[s_pmu], mb[s_pnu], nb[s]);
                    });
                math::sin_batch(buf, buf, n);
                detail::visit_plane(
                    l, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                        T const c = -b * buf[s];
                        fmu[s] += c;
                        fnu[s_pmu] += c;
                        fmu[s_pnu] -= c;
                        fnu[s] -= c;
                    });
            }
        }
    }

    // Fused total action + force: one angle fill per plane feeds a single
    // sincos pass whose cos half is reduced (action) and sin half scattered
    // (force) — instead of the two full plane sweeps `s_full` +
    // `compute_force` would do. Returns the action without updating the
    // `last_s_full` cache — cache semantics stay with `s_full`. Used by the
    // LLR WindowedAction, whose force scale needs S_base on every MD step.
    [[nodiscard]] double s_full_and_force(LinkLattice<T> const& l,
                                          LinkLattice<T>& force) const noexcept {
        std::fill(force.begin(), force.end(), T{0});
        std::size_t const d      = l.ndims();
        std::size_t const n      = l.nsites();
        std::size_t const n_plaq = (d * (d - 1) / 2) * n;
        double accum             = 0.0;
        T const b                = beta;
        ensure_scratch_(2 * n);
        T* const buf  = scratch.data();      // plaquette angles, then sin in place
        T* const cbuf = scratch.data() + n;  // cos
        for (std::size_t mu = 0; mu < d; ++mu) {
            T const* mb  = l.mu_data(mu);
            T* const fmu = force.mu_data(mu);
            for (std::size_t nu = mu + 1; nu < d; ++nu) {
                T const* nb  = l.mu_data(nu);
                T* const fnu = force.mu_data(nu);
                detail::visit_plane(
                    l, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                        buf[s] = detail::u1_plaq(mb[s], nb[s_pmu], mb[s_pnu], nb[s]);
                    });
                math::sincos_batch(buf, cbuf, buf, n);
                {
#if defined(__clang__)
                    _Pragma("clang fp reassociate(on)")
#endif
                        for (std::size_t s = 0; s < n; ++s) {
                        accum += static_cast<double>(cbuf[s]);
                    }
                }
                detail::visit_plane(
                    l, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                        T const c = -b * buf[s];
                        fmu[s] += c;
                        fnu[s_pmu] += c;
                        fmu[s_pnu] -= c;
                        fnu[s] -= c;
                    });
            }
        }
        return static_cast<double>(beta) * (static_cast<double>(n_plaq) - accum);
    }

    void
    compute_force_and_kick(LinkLattice<T> const& l, LinkLattice<T>& mom, T k_dt) const noexcept {
        std::size_t const d = l.ndims();
        std::size_t const n = l.nsites();
        T const c0          = -k_dt * beta;
        ensure_scratch_(n);
        T* const buf = scratch.data();
        for (std::size_t mu = 0; mu < d; ++mu) {
            T const* mb  = l.mu_data(mu);
            T* const mmu = mom.mu_data(mu);
            for (std::size_t nu = mu + 1; nu < d; ++nu) {
                T const* nb  = l.mu_data(nu);
                T* const mnu = mom.mu_data(nu);
                detail::visit_plane(
                    l, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                        buf[s] = detail::u1_plaq(mb[s], nb[s_pmu], mb[s_pnu], nb[s]);
                    });
                math::sin_batch(buf, buf, n);
                detail::visit_plane(
                    l, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                        T const c = c0 * buf[s];
                        mmu[s] += c;
                        mnu[s_pmu] += c;
                        mmu[s_pnu] -= c;
                        mnu[s] -= c;
                    });
            }
        }
    }

    // Per-plane plaquette / sin-plaq / cos-plaq scratch — populated by
    // vector libm at the start of each plane's loop. Sized lazily to nsites.
    mutable std::vector<T> scratch{};

    mutable double last_s_full_ = std::numeric_limits<double>::quiet_NaN();

private:
    void ensure_scratch_(std::size_t n) const noexcept {
        if (scratch.size() < n) {
            scratch.resize(n);
        }
    }
};

}  // namespace reticolo::action
