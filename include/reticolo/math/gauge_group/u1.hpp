#pragma once

#include <reticolo/math/gauge_group/base.hpp>

#include <cstddef>
#include <string_view>

namespace reticolo::gauge_group {

// U(1) gauge group model — CORE group operations only: the group constants and
// the HMC algebra hooks (`sample_algebra_slab` / `kinetic_slab` / `expi_lmul_slab`)
// the integrator calls. The Wilson plaquette physics (Re Tr U_p, the staple
// force) is action-specific and lives in `action/gauge/formula/wilson_u1.hpp`
// as `wilson_kernels<U1>`.
//
// Each link element is a phase U = exp(iθ) stored as one real angle θ, so
// n_real_components = 1, n_color = 1. Abelian: P ∈ iℝ is the real angle-velocity
// p, and the group exp degenerates to θ += dt·p.

struct U1 {
    using scalar_t                                 = double;
    static constexpr std::size_t n_real_components = 1;
    static constexpr std::size_t n_color           = 1;
    static constexpr std::string_view name         = "U1";

    // Sample p ~ N(0,1) so K = (1/2)·Σ p² gives detailed balance.
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
};

static_assert(GaugeGroup<U1>);

}  // namespace reticolo::gauge_group
