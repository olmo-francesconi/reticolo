#pragma once

#include <reticolo/core/indexing.hpp>
#include <reticolo/core/link_lattice.hpp>
#include <reticolo/core/site.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace reticolo::gauge::action {

// =============================================================================
//  Compact U(1) gauge theory with the Wilson plaquette action, written in
//  the paper convention of Langfeld-Lucini-Pellegrini-Rago (arxiv:1509.08391):
//
//      S = beta * sum_x  sum_{mu<nu}  cos( theta_{mu,nu}(x) )
//
//      theta_{mu,nu}(x) =  theta_mu(x)
//                        + theta_nu(x + mu_hat)
//                        - theta_mu(x + nu_hat)
//                        - theta_nu(x)
//
//  Boltzmann weight in this convention is exp(+S) (NOT exp(-S)). The gauge
//  HMC uses Hamiltonian H = K - S so that exp(-H) = exp(-K)*exp(+S). All
//  downstream sign conventions (compute_force, Metropolis acceptance,
//  WindowedAction) follow from this.
//
//  Force on each link (HMC drift `mom += k_dt * F` with F = +dS/dtheta):
//      F_mu(x)      <-  -beta * sin(theta_{mu,nu}(x))      (forward plaq)
//      F_nu(x+mu)   <-  -beta * sin(theta_{mu,nu}(x))
//      F_mu(x+nu)   <-  +beta * sin(theta_{mu,nu}(x))
//      F_nu(x)      <-  +beta * sin(theta_{mu,nu}(x))
//
//  (The numerical values coincide with the previous "1 - cos" convention's
//  -dS/dtheta — the two sign flips cancel — so the plaquette-centric scatter
//  code is unchanged; only the semantic interpretation differs.)
//
//  s_full and the force kernels are plaquette-centric: each plaquette is
//  evaluated exactly once and scatters its (beta * sin) contribution to the
//  four boundary links. Halves the trig-call count vs a link-centric loop.
//
//  s_local / ds_local on the Metropolis path return paper-convention values
//  (sums of `cos`, not `1 - cos`). The LinkMetropolis acceptance is
//  `accept if ds >= 0 or rng < exp(ds)` (paper convention favours moves
//  that increase S).
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
            // dS = beta * sum [cos(new) - cos(old)] over plaquettes touching the link.
            accum += std::cos(fwd_old + dtheta) - std::cos(fwd_old);
            accum += std::cos(bwd_old - dtheta) - std::cos(bwd_old);
        }
        return beta * accum;
    }

    // ---------- HasLinkSEff — plaquette-centric hot loop ----------

    [[nodiscard]] T s_full(LinkLattice<T> const& l) const noexcept {
        Indexing const& idx          = l.indexing_ref();
        Site::value_type const* next = idx.next_data();
        T const* tp                  = l.data();
        std::size_t const ns         = l.nsites();
        std::size_t const d          = l.ndims();
        T accum                      = T{0};
        for (std::size_t s = 0; s < ns; ++s) {
            std::size_t const base_s = s * d;
            for (std::size_t mu = 0; mu < d; ++mu) {
                std::size_t const base_pmu = next[base_s + mu] * d;
                T const t_mu_x             = tp[base_s + mu];
                for (std::size_t nu = mu + 1; nu < d; ++nu) {
                    std::size_t const base_pnu = next[base_s + nu] * d;
                    T const plaq = t_mu_x + tp[base_pmu + nu] - tp[base_pnu + mu] - tp[base_s + nu];
                    accum += std::cos(plaq);
                }
            }
        }
        return beta * accum;
    }

    // ---------- HasLinkForce — plaquette-centric scatter ----------

    void compute_force(LinkLattice<T> const& l, LinkLattice<T>& force) const noexcept {
        std::fill(force.begin(), force.end(), T{0});
        Indexing const& idx          = l.indexing_ref();
        Site::value_type const* next = idx.next_data();
        T const* tp                  = l.data();
        T* fp                        = force.data();
        std::size_t const ns         = l.nsites();
        std::size_t const d          = l.ndims();
        T const b                    = beta;
        for (std::size_t s = 0; s < ns; ++s) {
            std::size_t const base_s = s * d;
            for (std::size_t mu = 0; mu < d; ++mu) {
                std::size_t const base_pmu = next[base_s + mu] * d;
                T const t_mu_x             = tp[base_s + mu];
                for (std::size_t nu = mu + 1; nu < d; ++nu) {
                    std::size_t const base_pnu = next[base_s + nu] * d;
                    T const plaq = t_mu_x + tp[base_pmu + nu] - tp[base_pnu + mu] - tp[base_s + nu];
                    T const c    = -b * std::sin(plaq);
                    fp[base_s + mu] += c;
                    fp[base_pmu + nu] += c;
                    fp[base_pnu + mu] -= c;
                    fp[base_s + nu] -= c;
                }
            }
        }
    }

    // ---------- HasLinkFusedKick — same scatter into mom ----------

    void
    compute_force_and_kick(LinkLattice<T> const& l, LinkLattice<T>& mom, T k_dt) const noexcept {
        Indexing const& idx          = l.indexing_ref();
        Site::value_type const* next = idx.next_data();
        T const* tp                  = l.data();
        T* mp                        = mom.data();
        std::size_t const ns         = l.nsites();
        std::size_t const d          = l.ndims();
        T const c0                   = -k_dt * beta;
        for (std::size_t s = 0; s < ns; ++s) {
            std::size_t const base_s = s * d;
            for (std::size_t mu = 0; mu < d; ++mu) {
                std::size_t const base_pmu = next[base_s + mu] * d;
                T const t_mu_x             = tp[base_s + mu];
                for (std::size_t nu = mu + 1; nu < d; ++nu) {
                    std::size_t const base_pnu = next[base_s + nu] * d;
                    T const plaq = t_mu_x + tp[base_pmu + nu] - tp[base_pnu + mu] - tp[base_s + nu];
                    T const c    = c0 * std::sin(plaq);
                    mp[base_s + mu] += c;
                    mp[base_pmu + nu] += c;
                    mp[base_pnu + mu] -= c;
                    mp[base_s + nu] -= c;
                }
            }
        }
    }
};

}  // namespace reticolo::gauge::action
