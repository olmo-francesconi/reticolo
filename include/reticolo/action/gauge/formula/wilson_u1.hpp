#pragma once

#include <reticolo/action/gauge/detail/plane.hpp>
#include <reticolo/action/gauge/detail/wilson_kernels.hpp>
#include <reticolo/action/gauge/formula/wilson_u1_formula.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/math/gauge_group/u1.hpp>
#include <reticolo/math/vec_libm.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

// Wilson plaquette kernels for U(1) — the action-specific physics for the
// Wilson action on a `MatrixLinkLattice<U1, T>` (the abelian angle field). The
// double path batches four signed angles through a Sleef sin/cos scratch; the
// non-double path uses per-plaquette scalar libm. Bit-identical to CompactU1 on
// the same configuration (the shared design point tested in the physics suite).

namespace reticolo::action::detail {

namespace u1_detail {
// One nsites-sized plane scratch, owned per call-site via a thread_local so the
// kernels stay stateless.
[[gnu::always_inline]] inline double* plane_scratch(std::size_t n) noexcept {
    thread_local std::vector<double> buf;
    if (buf.size() < n) {
        buf.resize(n);
    }
    return buf.data();
}
}  // namespace u1_detail

template <>
struct wilson_kernels<gauge_group::U1> {
    using G = gauge_group::U1;

    // Re Tr U_p = cos(θ_μ(s) + θ_ν(s+pmu) − θ_μ(s+pnu) − θ_ν(s)). `stride` unused.
    template <class T>
    [[gnu::always_inline]] static inline double plaq_re_tr(T const* mb,
                                                           T const* nb,
                                                           std::size_t s,
                                                           std::size_t s_pmu,
                                                           std::size_t s_pnu,
                                                           std::size_t /*stride*/) noexcept {
        T const theta_p = u1_plaq(mb[s], nb[s_pmu], mb[s_pnu], nb[s]);
        return static_cast<double>(std::cos(theta_p));
    }

    // ∂(Re Tr U_p)/∂θ_μ(s) = −sin(θ_p); force F_μ(s) += −(β/N)·sin(θ_p), sign
    // flipping on the two "minus" links.
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
        T const theta_p = u1_plaq(mb[s], nb[s_pmu], mb[s_pnu], nb[s]);
        T const c       = static_cast<T>(-beta_over_n) * std::sin(theta_p);
        fmu[s] += c;
        fnu[s_pmu] += c;
        fmu[s_pnu] -= c;
        fnu[s] -= c;
    }

    // Plaquette-centric Wilson force scatter. Double path: stash plaquette angles
    // → Sleef sin_batch → scatter −(β/N)·sin(θ_p) into the 4 link forces.
    // Non-double falls back to per-plaquette scalar libm.
    template <class T>
    static void compute_force(MatrixLinkLattice<G, T> const& u,
                              MatrixLinkLattice<G, T>& force,
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
                    visit_plane(
                        u, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                            buf[s] =
                                u1_plaq(u_mu_blk[s], u_nu_blk[s_pmu], u_mu_blk[s_pnu], u_nu_blk[s]);
                        });
                    math::sin_batch(buf, buf, ns);
                    visit_plane(
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
                    visit_plane(
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

    // Fused force-and-kick: scatter −(k_dt·β/N)·sin(θ_p) directly into momentum.
    template <class T>
    static void compute_force_and_kick(MatrixLinkLattice<G, T> const& u,
                                       MatrixLinkLattice<G, T>& mom,
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
                    visit_plane(
                        u, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                            buf[s] =
                                u1_plaq(u_mu_blk[s], u_nu_blk[s_pmu], u_mu_blk[s_pnu], u_nu_blk[s]);
                        });
                    math::sin_batch(buf, buf, ns);
                    visit_plane(
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
                    visit_plane(
                        u, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                            T const theta_p =
                                u1_plaq(u_mu_blk[s], u_nu_blk[s_pmu], u_mu_blk[s_pnu], u_nu_blk[s]);
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

    // Plane-batched Σ Re Tr U_p over one (μ, ν) plane — the Wilson s_full fast
    // path (double-only Sleef cos).
    template <class T>
    static double s_full_plane_re_tr_sum(MatrixLinkLattice<G, T> const& u,
                                         std::size_t mu,
                                         std::size_t nu) noexcept
        requires std::is_same_v<T, double>
    {
        std::size_t const ns         = u.nsites();
        double* const buf            = u1_detail::plane_scratch(ns);
        double const* const u_mu_blk = u.mu_block_data(mu);
        double const* const u_nu_blk = u.mu_block_data(nu);
        visit_plane(u, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
            buf[s] = u1_plaq(u_mu_blk[s], u_nu_blk[s_pmu], u_mu_blk[s_pnu], u_nu_blk[s]);
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

    // Fused Σ Re Tr U_p + force in ONE sincos plane pass: the cos half is reduced
    // into the returned `accum_re_tr` and the sin half is scattered as
    // −(β/N)·sin(θ_p) into the four link forces. Returns the plaquette Re-Tr sum
    // (the caller — Wilson<U1>::s_full_and_force — applies the β·n_plaq combine);
    // the LLR WindowedAction needs S_base + force every MD step, and this is the
    // single-pass path CompactU1 used to provide. Double-only (Sleef sincos).
    template <class T>
    static double s_full_and_force(MatrixLinkLattice<G, T>& force_of,
                                   MatrixLinkLattice<G, T> const& u,
                                   double beta_over_n) noexcept {
        std::fill(force_of.begin(), force_of.end(), T{0});
        std::size_t const d  = u.ndims();
        std::size_t const ns = u.nsites();
        double accum         = 0.0;
        if constexpr (std::is_same_v<T, double>) {
            double* const buf  = u1_detail::plane_scratch(2 * ns);  // angles → sin in place
            double* const cbuf = buf + ns;                          // cos
            double const c0    = -beta_over_n;
            for (std::size_t mu = 0; mu < d; ++mu) {
                double const* const u_mu_blk = u.mu_block_data(mu);
                double* const f_mu_blk       = force_of.mu_block_data(mu);
                for (std::size_t nu = mu + 1; nu < d; ++nu) {
                    double const* const u_nu_blk = u.mu_block_data(nu);
                    double* const f_nu_blk       = force_of.mu_block_data(nu);
                    visit_plane(
                        u, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                            buf[s] =
                                u1_plaq(u_mu_blk[s], u_nu_blk[s_pmu], u_mu_blk[s_pnu], u_nu_blk[s]);
                        });
                    math::sincos_batch(buf, cbuf, buf, ns);
                    {
#if defined(__clang__)
                        _Pragma("clang fp reassociate(on)")
#endif
                            for (std::size_t s = 0; s < ns; ++s) {
                            accum += cbuf[s];
                        }
                    }
                    visit_plane(
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
            // Non-double: one fused pass with per-plaquette scalar sin/cos, matching
            // the plaq_re_tr (s_full) + plaq_force_accum (compute_force) fallbacks.
            T const c0_T = static_cast<T>(-beta_over_n);
            for (std::size_t mu = 0; mu < d; ++mu) {
                T const* const u_mu_blk = u.mu_block_data(mu);
                T* const f_mu_blk       = force_of.mu_block_data(mu);
                for (std::size_t nu = mu + 1; nu < d; ++nu) {
                    T const* const u_nu_blk = u.mu_block_data(nu);
                    T* const f_nu_blk       = force_of.mu_block_data(nu);
                    visit_plane(
                        u, mu, nu, [&](std::size_t s, std::size_t s_pmu, std::size_t s_pnu) {
                            T const theta_p =
                                u1_plaq(u_mu_blk[s], u_nu_blk[s_pmu], u_mu_blk[s_pnu], u_nu_blk[s]);
                            accum += static_cast<double>(std::cos(theta_p));
                            T const c = c0_T * std::sin(theta_p);
                            f_mu_blk[s] += c;
                            f_nu_blk[s_pmu] += c;
                            f_mu_blk[s_pnu] -= c;
                            f_nu_blk[s] -= c;
                        });
                }
            }
        }
        return accum;
    }
};

}  // namespace reticolo::action::detail
