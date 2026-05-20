#pragma once

#include <reticolo/action/detail/gauge_helpers.hpp>
#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/indexing.hpp>
#include <reticolo/core/link_lattice.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/math/vec_libm.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <vector>

namespace reticolo::action {

// =============================================================================
//  Compact U(1) gauge theory with the standard Wilson plaquette action:
//
//      S_W = beta * sum_x  sum_{mu<nu}  ( 1 - cos( theta_{mu,nu}(x) ) )
//
//      theta_{mu,nu}(x) =  theta_mu(x)
//                        + theta_nu(x + mu_hat)
//                        - theta_mu(x + nu_hat)
//                        - theta_nu(x)
//
//  This is the convention used by every production lattice code (openQCD,
//  Grid, MILC, Chroma, QUDA) and by Gattringer-Lang. S_W is bounded below
//  by 0 (cold config), reaches `beta * n_plaq` at maximal disorder.
//
//  Boltzmann weight: exp(-S_W). The gauge HMC uses H = K + S so
//  exp(-H) = exp(-K - S). `compute_force` returns the *force* F = -dS/dtheta;
//  the integrator kick is `mom += k_dt * F`. `LinkMetropolis` accepts moves
//  with `ds <= 0 || rng < exp(-ds)` (favours moves that decrease S).
//
//  The (1 - cos) constant `beta * n_plaq_per_link` and `beta * n_plaq_total`
//  on `s_local` / `s_full` makes both non-negative for sanity checking — the
//  HMC dynamics and Metropolis acceptance only depend on differences, where
//  the constant cancels.
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

    void describe(log::Entry& e) const {
        e.line("CompactU1<{}>", scalar_name<T>());
        e.param("β={:.3f}", beta);
    }

    // ---------- helpers ----------

    [[nodiscard]] T
    plaq_angle(LinkLattice<T> const& l, Site x, std::size_t mu, std::size_t nu) const noexcept {
        Site const x_pmu = l.next(x, mu);
        Site const x_pnu = l.next(x, nu);
        return l(x, mu) + l(x_pmu, nu) - l(x_pnu, mu) - l(x, nu);
    }

    // ---------- LinkLocalAction (Metropolis path) ----------

    [[nodiscard]] T s_local(LinkLattice<T> const& l, Site x, std::size_t mu) const noexcept {
        T cos_sum           = T{0};
        std::size_t const d = l.ndims();
        for (std::size_t nu = 0; nu < d; ++nu) {
            if (nu == mu) {
                continue;
            }
            Site const x_mnu     = l.prev(x, nu);
            Site const x_mnu_pmu = l.next(x_mnu, mu);
            T const fwd          = plaq_angle(l, x, mu, nu);
            T const bwd          = l(x_mnu, mu) + l(x_mnu_pmu, nu) - l(x, mu) - l(x_mnu, nu);
            cos_sum += std::cos(fwd) + std::cos(bwd);
        }
        // S contribution from the 2(d-1) plaquettes through link (x, mu),
        // standard Wilson form: beta * (n_plaq_per_link - sum cos).
        T const n_plaq_per_link = T{2} * static_cast<T>(d - 1);
        return beta * (n_plaq_per_link - cos_sum);
    }

    [[nodiscard]] T
    ds_local(LinkLattice<T> const& l, Site x, std::size_t mu, T new_v) const noexcept {
        T const dtheta      = new_v - l(x, mu);
        T cos_delta         = T{0};
        std::size_t const d = l.ndims();
        for (std::size_t nu = 0; nu < d; ++nu) {
            if (nu == mu) {
                continue;
            }
            Site const x_mnu     = l.prev(x, nu);
            Site const x_mnu_pmu = l.next(x_mnu, mu);
            T const fwd_old      = plaq_angle(l, x, mu, nu);
            T const bwd_old      = l(x_mnu, mu) + l(x_mnu_pmu, nu) - l(x, mu) - l(x_mnu, nu);
            cos_delta += std::cos(fwd_old + dtheta) - std::cos(fwd_old);
            cos_delta += std::cos(bwd_old - dtheta) - std::cos(bwd_old);
        }
        // d/dt (sum 1-cos) = -d/dt (sum cos), so flip sign.
        return -beta * cos_delta;
    }

    // ---------- HasLinkSEff — plane-by-plane on direction-major blocks ------

    [[nodiscard]] T s_full(LinkLattice<T> const& l) const noexcept {
        std::size_t const d      = l.ndims();
        std::size_t const n_plaq = (d * (d - 1) / 2) * l.nsites();
        T accum                  = T{0};
        if constexpr (std::is_same_v<T, double>) {
            std::size_t const n = l.nsites();
            ensure_scratch_(n);
            double* const buf = scratch.data();
            for (std::size_t mu = 0; mu < d; ++mu) {
                T const* mb = l.mu_data(mu);
                for (std::size_t nu = mu + 1; nu < d; ++nu) {
                    T const* nb = l.mu_data(nu);
                    // Pass 1: stash plaquette angles for the plane into the scratch.
                    detail::visit_plane(
                        l, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                            buf[s] = mb[s] + nb[s_pmu] - mb[s_pnu] - nb[s];
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
                            accum += buf[s];
                        }
                    }
                }
            }
        } else {
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
        }
        // Standard Wilson form: S = beta * sum (1 - cos theta_p).
        T const s    = beta * (static_cast<T>(n_plaq) - accum);
        last_s_full_ = s;
        return s;
    }

    [[nodiscard]] T last_s_full() const noexcept { return last_s_full_; }
    void restore_last_s_full(T v) const noexcept { last_s_full_ = v; }

    // ---------- HasLinkForce — plaquette-centric scatter -------------------

    void compute_force(LinkLattice<T> const& l, LinkLattice<T>& force) const noexcept {
        std::fill(force.begin(), force.end(), T{0});
        std::size_t const d = l.ndims();
        T const b           = beta;
        if constexpr (std::is_same_v<T, double>) {
            std::size_t const n = l.nsites();
            ensure_scratch_(n);
            double* const buf = scratch.data();
            for (std::size_t mu = 0; mu < d; ++mu) {
                T const* mb  = l.mu_data(mu);
                T* const fmu = force.mu_data(mu);
                for (std::size_t nu = mu + 1; nu < d; ++nu) {
                    T const* nb  = l.mu_data(nu);
                    T* const fnu = force.mu_data(nu);
                    detail::visit_plane(
                        l, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                            buf[s] = mb[s] + nb[s_pmu] - mb[s_pnu] - nb[s];
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
        } else {
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
    }

    // ---------- HasLinkFusedKick — scatter directly into mom ---------------

    void
    compute_force_and_kick(LinkLattice<T> const& l, LinkLattice<T>& mom, T k_dt) const noexcept {
        std::size_t const d = l.ndims();
        T const c0          = -k_dt * beta;
        if constexpr (std::is_same_v<T, double>) {
            std::size_t const n = l.nsites();
            ensure_scratch_(n);
            double* const buf = scratch.data();
            for (std::size_t mu = 0; mu < d; ++mu) {
                T const* mb  = l.mu_data(mu);
                T* const mmu = mom.mu_data(mu);
                for (std::size_t nu = mu + 1; nu < d; ++nu) {
                    T const* nb  = l.mu_data(nu);
                    T* const mnu = mom.mu_data(nu);
                    detail::visit_plane(
                        l, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                            buf[s] = mb[s] + nb[s_pmu] - mb[s_pnu] - nb[s];
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
        } else {
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
    }

    // Per-plane plaquette / sin-plaq / cos-plaq scratch — populated by
    // vector libm at the start of each plane's loop. Sized lazily to nsites.
    mutable std::vector<double> scratch{};

    mutable T last_s_full_ = std::numeric_limits<T>::quiet_NaN();

private:
    void ensure_scratch_(std::size_t n) const noexcept {
        if (scratch.size() < n) {
            scratch.resize(n);
        }
    }
};

}  // namespace reticolo::action
