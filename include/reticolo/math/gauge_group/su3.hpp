#pragma once

#include <reticolo/math/gauge_group/base.hpp>
#include <reticolo/math/su3_ops.hpp>

#include <cstddef>
#include <string_view>

namespace reticolo::gauge_group {

// SU(3) gauge group model — CORE group operations only: the constants and the
// HMC algebra hooks (thin wrappers over math::su3). The Wilson plaquette physics
// (Re Tr U_p, the staple force) is action-specific — see
// action/gauge/formula/wilson_su3.hpp (`wilson_kernels<SU3>`).
//
// Storage: full 3×3 complex matrix, 18 reals per link (see math::su3). N = 3.

struct SU3 {
    using scalar_t                                 = double;
    static constexpr std::size_t n_real_components = 18;
    static constexpr std::size_t n_color           = 3;
    static constexpr std::string_view name         = "SU3";

    template <class T, class Rng>
    [[gnu::always_inline]] static inline void
    sample_algebra_slab(T* p_blk, Rng& rng, std::size_t n) noexcept {
        math::su3::sample_algebra_slab(p_blk, rng, n);
    }

    template <class T>
    [[gnu::always_inline]] static inline double kinetic_slab(T const* p_blk,
                                                             std::size_t n) noexcept {
        return math::su3::kinetic_slab(p_blk, n);
    }

    template <class T>
    [[gnu::always_inline]] static inline void
    expi_lmul_slab(T* u_blk, T const* p_blk, double dt, std::size_t n) noexcept {
        math::su3::expi_lmul_slab(u_blk, p_blk, dt, n);
    }
};

static_assert(GaugeGroup<SU3>);

}  // namespace reticolo::gauge_group
