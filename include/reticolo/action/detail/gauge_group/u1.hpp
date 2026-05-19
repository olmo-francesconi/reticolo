#pragma once

#include <reticolo/action/detail/gauge_group/base.hpp>
#include <reticolo/action/detail/gauge_helpers.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/math/vec_libm.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace reticolo::gauge_group {

// =============================================================================
//  U(1) gauge group model. Each link element is a phase U = exp(iθ) stored
//  as a single real angle θ ∈ ℝ — so n_real_components = 1. The group is
//  abelian, so the plaquette product reduces to a sum of four signed angles
//  and Re Tr U_p = cos(θ_p). All "matrix" math collapses to scalar math.
//
//  The slab-batched hot paths here use the same Sleef cos/sin scratch
//  pattern as `action::CompactU1`, so Wilson<U(1)> matches CompactU1
//  throughput on the HMC bench (M8: retire CompactU1 once verified).
//  Scratch is owned per call-site via a function-local thread_local —
//  the gauge group struct itself stays stateless.
// =============================================================================
//
// One nsites-sized scratch shared by all hot paths in this TU.
namespace u1_detail {
[[gnu::always_inline]] inline double* plane_scratch(std::size_t n) noexcept {
    thread_local std::vector<double> buf;
    if (buf.size() < n) {
        buf.resize(n);
    }
    return buf.data();
}
}  // namespace u1_detail

struct U1 {
    using scalar_t                                 = double;
    static constexpr std::size_t n_real_components = 1;
    static constexpr std::size_t n_color           = 1;

    // Re Tr U_p = cos(theta_mu(s) + theta_nu(s+pmu) − theta_mu(s+pnu) − theta_nu(s)).
    // `stride` is unused for U(1) (the single component lives directly under
    // the base pointer at offset s).
    template <class T>
    [[gnu::always_inline]] static inline double plaq_re_tr(T const* mb,
                                                           T const* nb,
                                                           std::size_t s,
                                                           std::size_t s_pmu,
                                                           std::size_t s_pnu,
                                                           std::size_t /*stride*/) noexcept {
        T const theta_p = mb[s] + nb[s_pmu] - mb[s_pnu] - nb[s];
        return static_cast<double>(std::cos(theta_p));
    }

    // ∂(Re Tr U_p)/∂θ_μ(s) = −sin(θ_p); the Wilson force is
    //   F_μ(s) += −(β/N)·∂S/∂θ = (β/N)·∂(Re Tr U_p)/∂θ = −(β/N)·sin(θ_p)
    // and its sign flips on the two "minus" links (μ at s+ν̂ and ν at s).
    template <class T>
    [[gnu::always_inline]] static inline void plaq_force_accum(T const* mb,
                                                               T const* nb,
                                                               T* fmu,
                                                               T* fnu,
                                                               std::size_t s,
                                                               std::size_t s_pmu,
                                                               std::size_t s_pnu,
                                                               std::size_t /*stride*/,
                                                               double beta_over_n) noexcept {
        T const theta_p = mb[s] + nb[s_pmu] - mb[s_pnu] - nb[s];
        T const c       = static_cast<T>(-beta_over_n) * std::sin(theta_p);
        fmu[s] += c;
        fnu[s_pmu] += c;
        fmu[s_pnu] -= c;
        fnu[s] -= c;
    }

    // -------- HMC slab hooks --------------------------------------------------
    // U(1) is abelian: P ∈ iℝ stored as the real angle-velocity p; the group
    // exp degenerates to addition. Convention matches scalar HMC — sample
    // p ~ N(0, 1) so K = (1/2)·sum p² gives detailed balance.
    template <class Rng>
    [[gnu::always_inline]] static inline void
    sample_algebra_slab(double* p_blk, Rng& rng, std::size_t n) noexcept {
        for (std::size_t s = 0; s < n; ++s) {
            p_blk[s] = rng.normal();
        }
    }

    [[gnu::always_inline]] static inline double kinetic_slab(double const* p_blk,
                                                             std::size_t n) noexcept {
        double k = 0.0;
        for (std::size_t s = 0; s < n; ++s) {
            k += p_blk[s] * p_blk[s];
        }
        return 0.5 * k;
    }

    // U(1) drift: U_new = exp(i·dt·p)·U_old reduces to θ_new = θ_old + dt·p.
    [[gnu::always_inline]] static inline void
    expi_lmul_slab(double* u_blk, double const* p_blk, double dt, std::size_t n) noexcept {
        for (std::size_t s = 0; s < n; ++s) {
            u_blk[s] += dt * p_blk[s];
        }
    }

    // Plaquette-centric Wilson force scatter. The double specialisation goes
    // through a plane-batched Sleef sin path (same shape as CompactU1):
    // stash plaquette angles in nsites scratch → sin_batch → scatter the
    // −(β/N)·sin(θ_p) into the 4 link forces. Non-double T falls back to the
    // per-plaquette `plaq_force_accum` scalar libm path so float / extended
    // precision keep working without Sleef widths for them.
    template <class T>
    static void compute_force(MatrixLinkLattice<U1, T> const& u,
                              MatrixLinkLattice<U1, T>& force,
                              double beta_over_n) noexcept {
        std::fill(force.begin(), force.end(), T{0});
        std::size_t const d  = u.ndims();
        std::size_t const ns = u.nsites();
        if constexpr (std::is_same_v<T, double>) {
            double* const buf = u1_detail::plane_scratch(ns);
            double const c0   = -beta_over_n;
            for (std::size_t mu = 0; mu < d; ++mu) {
                double const* const u_mu_blk = u.mu_block_data(mu);
                double* const f_mu_blk       = force.mu_block_data(mu);
                for (std::size_t nu = mu + 1; nu < d; ++nu) {
                    double const* const u_nu_blk = u.mu_block_data(nu);
                    double* const f_nu_blk       = force.mu_block_data(nu);
                    action::detail::visit_plane(
                        u, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                            buf[s] = u_mu_blk[s] + u_nu_blk[s_pmu] - u_mu_blk[s_pnu] - u_nu_blk[s];
                        });
                    math::sin_batch(buf, buf, ns);
                    action::detail::visit_plane(
                        u, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                            double const c = c0 * buf[s];
                            f_mu_blk[s] += c;
                            f_nu_blk[s_pmu] += c;
                            f_mu_blk[s_pnu] -= c;
                            f_nu_blk[s] -= c;
                        });
                }
            }
        } else {
            for (std::size_t mu = 0; mu < d; ++mu) {
                T const* const u_mu_blk = u.mu_block_data(mu);
                T* const f_mu_blk       = force.mu_block_data(mu);
                for (std::size_t nu = mu + 1; nu < d; ++nu) {
                    T const* const u_nu_blk = u.mu_block_data(nu);
                    T* const f_nu_blk       = force.mu_block_data(nu);
                    action::detail::visit_plane(
                        u, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                            plaq_force_accum(u_mu_blk,
                                             u_nu_blk,
                                             f_mu_blk,
                                             f_nu_blk,
                                             s,
                                             s_pmu,
                                             s_pnu,
                                             ns,
                                             beta_over_n);
                        });
                }
            }
        }
    }

    // Fused force-and-kick: scatter −(k_dt·β/N)·sin(θ_p) directly into the
    // momentum LinkLattice. Skips the dedicated force buffer the integrator
    // would otherwise materialise via compute_force + kick_add. Double-only
    // batched path; non-double falls back to per-plaquette scalar sin.
    template <class T>
    static void compute_force_and_kick(MatrixLinkLattice<U1, T> const& u,
                                       MatrixLinkLattice<U1, T>& mom,
                                       double beta_over_n,
                                       double k_dt) noexcept {
        std::size_t const d  = u.ndims();
        std::size_t const ns = u.nsites();
        if constexpr (std::is_same_v<T, double>) {
            double* const buf = u1_detail::plane_scratch(ns);
            double const c0   = -k_dt * beta_over_n;
            for (std::size_t mu = 0; mu < d; ++mu) {
                double const* const u_mu_blk = u.mu_block_data(mu);
                double* const m_mu_blk       = mom.mu_block_data(mu);
                for (std::size_t nu = mu + 1; nu < d; ++nu) {
                    double const* const u_nu_blk = u.mu_block_data(nu);
                    double* const m_nu_blk       = mom.mu_block_data(nu);
                    action::detail::visit_plane(
                        u, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                            buf[s] = u_mu_blk[s] + u_nu_blk[s_pmu] - u_mu_blk[s_pnu] - u_nu_blk[s];
                        });
                    math::sin_batch(buf, buf, ns);
                    action::detail::visit_plane(
                        u, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                            double const c = c0 * buf[s];
                            m_mu_blk[s] += c;
                            m_nu_blk[s_pmu] += c;
                            m_mu_blk[s_pnu] -= c;
                            m_nu_blk[s] -= c;
                        });
                }
            }
        } else {
            T const c0_T = static_cast<T>(-k_dt * beta_over_n);
            for (std::size_t mu = 0; mu < d; ++mu) {
                T const* const u_mu_blk = u.mu_block_data(mu);
                T* const m_mu_blk       = mom.mu_block_data(mu);
                for (std::size_t nu = mu + 1; nu < d; ++nu) {
                    T const* const u_nu_blk = u.mu_block_data(nu);
                    T* const m_nu_blk       = mom.mu_block_data(nu);
                    action::detail::visit_plane(
                        u, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                            T const theta_p =
                                u_mu_blk[s] + u_nu_blk[s_pmu] - u_mu_blk[s_pnu] - u_nu_blk[s];
                            T const c = c0_T * std::sin(theta_p);
                            m_mu_blk[s] += c;
                            m_nu_blk[s_pmu] += c;
                            m_mu_blk[s_pnu] -= c;
                            m_nu_blk[s] -= c;
                        });
                }
            }
        }
    }

    // Plane-batched Σ Re Tr U_p over a single (mu, nu) plane. Returned value
    // is added to an external accumulator by Wilson<G>::s_full. Double-only
    // batched path; for other T the caller falls back to per-plaquette
    // `plaq_re_tr`.
    template <class T>
    static double s_full_plane_re_tr_sum(MatrixLinkLattice<U1, T> const& u,
                                         std::size_t mu,
                                         std::size_t nu) noexcept
        requires std::is_same_v<T, double>
    {
        std::size_t const ns         = u.nsites();
        double* const buf            = u1_detail::plane_scratch(ns);
        double const* const u_mu_blk = u.mu_block_data(mu);
        double const* const u_nu_blk = u.mu_block_data(nu);
        action::detail::visit_plane(
            u, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                buf[s] = u_mu_blk[s] + u_nu_blk[s_pmu] - u_mu_blk[s_pnu] - u_nu_blk[s];
            });
        math::cos_batch(buf, buf, ns);
        double accum = 0.0;
        {
#if defined(__clang__)
            _Pragma("clang fp reassociate(on)")
#endif
                for (std::size_t s = 0; s < ns; ++s) {
                accum += buf[s];
            }
        }
        return accum;
    }
};

static_assert(GaugeGroup<U1>);

}  // namespace reticolo::gauge_group
