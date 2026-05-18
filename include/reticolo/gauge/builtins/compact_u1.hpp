#pragma once

#include <reticolo/core/indexing.hpp>
#include <reticolo/core/link_lattice.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/gauge/hot_loop.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace reticolo::gauge::action {

// =============================================================================
//  Compact U(1) gauge theory with the Wilson plaquette action, in the paper
//  convention of Langfeld-Lucini-Pellegrini-Rago (arxiv:1509.08391):
//
//      S = beta * sum_x  sum_{mu<nu}  cos( theta_{mu,nu}(x) )
//
//      theta_{mu,nu}(x) =  theta_mu(x)
//                        + theta_nu(x + mu_hat)
//                        - theta_mu(x + nu_hat)
//                        - theta_nu(x)
//
//  Boltzmann weight in this convention is exp(+S) (NOT exp(-S)). The gauge
//  HMC uses Hamiltonian H = K - S so that exp(-H) = exp(-K) * exp(+S).
//  `compute_force` returns +dS/dtheta; the integrator kick is
//  `mom += k_dt * compute_force(field)`. `LinkMetropolis` accepts moves with
//  `ds >= 0 || rng < exp(ds)` (favours moves that increase S).
//
//  Storage layout: `LinkLattice<T>` is direction-major (each direction is a
//  contiguous nsites-element block). The hot loops here iterate one
//  plaquette plane (mu, nu) at a time via `gauge::detail::visit_plane`,
//  which peels wrap slabs off the boundary so the bulk body sees stride-1
//  reads and writes through fixed `(s_pmu - s, s_pnu - s)` offsets — clean
//  autovectorisation territory.
// =============================================================================

template <class T = double>
struct CompactU1 {
    using value_type = T;
    using field_type = LinkLattice<T>;

    T beta = T{0};

    // ---------- helpers ----------

    [[nodiscard]] T
    plaq_angle(LinkLattice<T> const& l, Site x, std::size_t mu, std::size_t nu) const noexcept {
        Site const x_pmu = l.next(x, mu);
        Site const x_pnu = l.next(x, nu);
        return l(x, mu) + l(x_pmu, nu) - l(x_pnu, mu) - l(x, nu);
    }

    // ---------- LinkLocalAction (Metropolis path) ----------

    [[nodiscard]] T s_local(LinkLattice<T> const& l, Site x, std::size_t mu) const noexcept {
        T accum             = T{0};
        std::size_t const d = l.ndims();
        for (std::size_t nu = 0; nu < d; ++nu) {
            if (nu == mu) {
                continue;
            }
            Site const x_mnu     = l.prev(x, nu);
            Site const x_mnu_pmu = l.next(x_mnu, mu);
            T const fwd          = plaq_angle(l, x, mu, nu);
            T const bwd          = l(x_mnu, mu) + l(x_mnu_pmu, nu) - l(x, mu) - l(x_mnu, nu);
            accum += std::cos(fwd) + std::cos(bwd);
        }
        return beta * accum;
    }

    [[nodiscard]] T
    ds_local(LinkLattice<T> const& l, Site x, std::size_t mu, T new_v) const noexcept {
        T const dtheta      = new_v - l(x, mu);
        T accum             = T{0};
        std::size_t const d = l.ndims();
        for (std::size_t nu = 0; nu < d; ++nu) {
            if (nu == mu) {
                continue;
            }
            Site const x_mnu     = l.prev(x, nu);
            Site const x_mnu_pmu = l.next(x_mnu, mu);
            T const fwd_old      = plaq_angle(l, x, mu, nu);
            T const bwd_old      = l(x_mnu, mu) + l(x_mnu_pmu, nu) - l(x, mu) - l(x_mnu, nu);
            accum += std::cos(fwd_old + dtheta) - std::cos(fwd_old);
            accum += std::cos(bwd_old - dtheta) - std::cos(bwd_old);
        }
        return beta * accum;
    }

    // ---------- HasLinkSEff — plane-by-plane on direction-major blocks ------

    [[nodiscard]] T s_full(LinkLattice<T> const& l) const noexcept {
        std::size_t const d = l.ndims();
        T accum             = T{0};
        for (std::size_t mu = 0; mu < d; ++mu) {
            T const* mb = l.mu_data(mu);
            for (std::size_t nu = mu + 1; nu < d; ++nu) {
                T const* nb = l.mu_data(nu);
                detail::visit_plane(
                    l, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                        T const plaq = mb[s] + nb[s_pmu] - mb[s_pnu] - nb[s];
                        accum += std::cos(plaq);
                    });
            }
        }
        return beta * accum;
    }

    // ---------- HasLinkForce — plaquette-centric scatter -------------------

    void compute_force(LinkLattice<T> const& l, LinkLattice<T>& force) const noexcept {
        std::fill(force.begin(), force.end(), T{0});
        std::size_t const d = l.ndims();
        T const b           = beta;
        for (std::size_t mu = 0; mu < d; ++mu) {
            T const* mb  = l.mu_data(mu);
            T* const fmu = force.mu_data(mu);
            for (std::size_t nu = mu + 1; nu < d; ++nu) {
                T const* nb  = l.mu_data(nu);
                T* const fnu = force.mu_data(nu);
                detail::visit_plane(
                    l, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                        T const plaq = mb[s] + nb[s_pmu] - mb[s_pnu] - nb[s];
                        T const c    = -b * std::sin(plaq);
                        fmu[s] += c;
                        fnu[s_pmu] += c;
                        fmu[s_pnu] -= c;
                        fnu[s] -= c;
                    });
            }
        }
    }

    // ---------- HasLinkFusedKick — scatter directly into mom ---------------

    void
    compute_force_and_kick(LinkLattice<T> const& l, LinkLattice<T>& mom, T k_dt) const noexcept {
        std::size_t const d = l.ndims();
        T const c0          = -k_dt * beta;
        for (std::size_t mu = 0; mu < d; ++mu) {
            T const* mb  = l.mu_data(mu);
            T* const mmu = mom.mu_data(mu);
            for (std::size_t nu = mu + 1; nu < d; ++nu) {
                T const* nb  = l.mu_data(nu);
                T* const mnu = mom.mu_data(nu);
                detail::visit_plane(
                    l, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                        T const plaq = mb[s] + nb[s_pmu] - mb[s_pnu] - nb[s];
                        T const c    = c0 * std::sin(plaq);
                        mmu[s] += c;
                        mnu[s_pmu] += c;
                        mmu[s_pnu] -= c;
                        mnu[s] -= c;
                    });
            }
        }
    }
};

}  // namespace reticolo::gauge::action
