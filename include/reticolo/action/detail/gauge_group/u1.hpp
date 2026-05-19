#pragma once

#include <reticolo/action/detail/gauge_group/base.hpp>
#include <reticolo/action/detail/gauge_helpers.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace reticolo::gauge_group {

// =============================================================================
//  U(1) gauge group model. Each link element is a phase U = exp(iθ) stored
//  as a single real angle θ ∈ ℝ — so n_real_components = 1. The group is
//  abelian, so the plaquette product reduces to a sum of four signed angles
//  and Re Tr U_p = cos(θ_p). All "matrix" math collapses to scalar math.
//
//  The plaquette-action / plaquette-force kernels here are bitwise-identical
//  to the per-site fallback path in `action::CompactU1`. The vector-libm
//  batched path (Sleef cos/sin on scratch slabs) lives in CompactU1; once
//  Wilson<U1> ships an equivalent batched primitive we can retire CompactU1
//  (the M8 commit).
// =============================================================================

struct U1 {
    using scalar_t                                 = double;
    static constexpr std::size_t n_real_components = 1;
    static constexpr std::size_t n_color           = 1;

    // Re Tr U_p = cos(theta_mu(s) + theta_nu(s+pmu) − theta_mu(s+pnu) − theta_nu(s)).
    // `stride` is unused for U(1) (the single component lives directly under
    // the base pointer at offset s).
    template <class T>
    [[gnu::always_inline]] static inline double
    plaq_re_tr(T const* mb,
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
    [[gnu::always_inline]] static inline void
    plaq_force_accum(T const* mb,
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

    // Plaquette-centric Wilson force scatter — same shape as the legacy
    // `CompactU1::compute_force` per-plaquette path. For U(1) the force on a
    // link is just −β·sin(θ_p) added/subtracted into the 4 links of every
    // plaquette through it.
    template <class T>
    static void compute_force(MatrixLinkLattice<U1, T> const& u,
                              MatrixLinkLattice<U1, T>& force,
                              double beta_over_n) noexcept {
        std::fill(force.begin(), force.end(), T{0});
        std::size_t const d  = u.ndims();
        std::size_t const ns = u.nsites();
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
};

static_assert(GaugeGroup<U1>);

}  // namespace reticolo::gauge_group
