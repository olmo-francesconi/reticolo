#pragma once

#include <reticolo/action/detail/gauge_group/base.hpp>
#include <reticolo/action/detail/gauge_helpers.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/site.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>

namespace reticolo::action {

// Generic Wilson plaquette action for any SU(N) (and U(1)) gauge group
// conforming to the `gauge_group::GaugeGroup` concept.
//
//     S_W = (β/N) · Σ_x Σ_{μ<ν} ( N − Re Tr U_{μν}(x) )
//         = β · n_plaq − (β/N) · Σ_p Re Tr U_p
//
// The hot loops walk one plaquette plane (μ, ν) at a time via the existing
// `detail::visit_plane` driver (bulk-vs-slab, stride-1 in the inner site
// loop, peeled wrap boundaries). The per-plaquette math — Re Tr U_p and
// the staple force scatter — is delegated to the group model G, which
// receives base pointers to the nc-component link blocks for the two
// directions in the plane plus the three site indices `(s, s_pmu, s_pnu)`
// and the per-direction stride (= nsites). Models with `n_real_components
// = 1` (U(1)) ignore the stride; matrix groups index per component as
//     block_ptr[k * stride + s].
//
// At N=1 with U=exp(iθ) this reduces line-for-line to `CompactU1` — the
// bit-identity is the design point of M3 and lets the gauge group concept
// be validated on a known-correct path before the SU(2)/SU(3) instances
// arrive in M5/M7.

template <gauge_group::GaugeGroup G, class T = double>
struct Wilson {
    using value_type = T;
    using group_type = G;
    using field_type = MatrixLinkLattice<G, T>;

    T beta = T{0};

    // Uniform announce hook (called by reticolo::log::act): concept name +
    // gauge group, then the coupling on its own indented line.
    void describe(log::Entry& e) const {
        e.line("Wilson<{}>", G::name);
        e.param("β={:.3f}", beta);
    }

    // Returns double regardless of the field precision T: the plaquette sum is
    // accumulated in double, and the final β·n_plaq − (β/N)·Σ combine — a
    // difference of two ~volume-sized quantities — is also done in double to
    // avoid catastrophic cancellation when T = float.
    [[nodiscard]] double s_full(field_type const& U) const noexcept {
        std::size_t const d      = U.ndims();
        std::size_t const ns     = U.nsites();
        std::size_t const n_plaq = (d * (d - 1) / 2) * ns;
        double accum_re_tr       = 0.0;
        // If G provides a per-plane batched Σ Re Tr U_p, use it (Sleef cos
        // path on U(1)); otherwise fall back to per-plaquette plaq_re_tr.
        if constexpr (requires { G::template s_full_plane_re_tr_sum<T>(U, 0, 1); }) {
            for (std::size_t mu = 0; mu < d; ++mu) {
                for (std::size_t nu = mu + 1; nu < d; ++nu) {
                    accum_re_tr += G::template s_full_plane_re_tr_sum<T>(U, mu, nu);
                }
            }
        } else {
            for (std::size_t mu = 0; mu < d; ++mu) {
                T const* mb = U.mu_block_data(mu);
                for (std::size_t nu = mu + 1; nu < d; ++nu) {
                    T const* nb = U.mu_block_data(nu);
                    detail::visit_plane(
                        U, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                            accum_re_tr += G::plaq_re_tr(mb, nb, s, s_pmu, s_pnu, ns);
                        });
                }
            }
        }
        double const beta_d      = static_cast<double>(beta);
        double const beta_over_n = beta_d / static_cast<double>(G::n_color);
        double const s = (beta_d * static_cast<double>(n_plaq)) - (beta_over_n * accum_re_tr);
        last_s_full_   = s;
        return s;
    }

    // Raw-action cache: last `s_full` evaluated on the field. NaN before the
    // first call. The LLR Replica reads this after a trajectory instead of
    // re-sweeping; HMC snapshots and restores it across reject rollbacks.
    [[nodiscard]] double last_s_full() const noexcept { return last_s_full_; }
    void restore_last_s_full(double v) const noexcept { last_s_full_ = v; }

    // Each gauge group owns its force algorithm — U(1) uses the per-plaquette
    // scatter that matches CompactU1 bit-for-bit; SU(N) uses a link-centric
    // staple sum + TA[U·V] step that fits the matrix structure. Wilson<G> is
    // just a thin wrapper that hands G the beta/N prefactor.

    void compute_force(field_type const& U, field_type& force) const noexcept {
        T const beta_over_n          = beta / static_cast<T>(G::n_color);
        double const beta_over_n_dbl = static_cast<double>(beta_over_n);
        G::compute_force(U, force, beta_over_n_dbl);
    }

    // Fused force-and-kick: scatter the per-plaquette force contribution
    // directly into the momentum field with the integrator's scale, skipping
    // the dedicated force buffer the kick_add path would otherwise allocate
    // and stream. Only enabled when G provides its own fused kernel — that's
    // U(1) today; SU(2)/SU(3) plug in via the H1 SU(N) slab work.
    void compute_force_and_kick(field_type const& U, field_type& mom, T k_dt) const noexcept
        requires requires(field_type const& cu, field_type& m) {
            G::compute_force_and_kick(cu, m, double{}, double{});
        }
    {
        double const beta_over_n_dbl = static_cast<double>(beta / static_cast<T>(G::n_color));
        G::compute_force_and_kick(U, mom, beta_over_n_dbl, static_cast<double>(k_dt));
    }

    mutable double last_s_full_ = std::numeric_limits<double>::quiet_NaN();

    // Sum of Re Tr U_p over the 2(d−1) plaquettes through link (x, μ),
    // wrapped into the Wilson constant offset so the returned value is
    // non-negative for the cold config (theta = 0 → cos(0) = 1 → S_local = 0).

    [[nodiscard]] T s_local(field_type const& U, Site x, std::size_t mu) const noexcept {
        std::size_t const d  = U.ndims();
        std::size_t const ns = U.nsites();
        T const* mb          = U.mu_block_data(mu);
        double cos_sum       = 0.0;
        for (std::size_t nu = 0; nu < d; ++nu) {
            if (nu == mu) {
                continue;
            }
            T const* nb              = U.mu_block_data(nu);
            std::size_t const x_v    = x.value();
            std::size_t const x_pmu  = U.next(x, mu).value();
            std::size_t const x_pnu  = U.next(x, nu).value();
            Site const x_mnu_site    = U.prev(x, nu);
            std::size_t const x_mnu  = x_mnu_site.value();
            std::size_t const x_mnup = U.next(x_mnu_site, mu).value();
            // Forward plaquette anchored at x.
            cos_sum += G::plaq_re_tr(mb, nb, x_v, x_pmu, x_pnu, ns);
            // Backward plaquette anchored at x − ν̂.
            cos_sum += G::plaq_re_tr(mb, nb, x_mnu, x_mnup, x_v, ns);
        }
        T const n_plaq_per_link = T{2} * static_cast<T>(d - 1);
        T const N_re            = static_cast<T>(G::n_color);
        T const beta_over_n     = beta / N_re;
        return (beta * n_plaq_per_link) - (beta_over_n * static_cast<T>(cos_sum));
    }
};

}  // namespace reticolo::action
