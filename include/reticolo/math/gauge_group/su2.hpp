#pragma once

#include <reticolo/math/gauge_group/base.hpp>
#include <reticolo/math/su2_ops.hpp>

#include <cstddef>
#include <string_view>

namespace reticolo::gauge_group {

// SU(2) gauge group model — CORE group operations only: the constants and the
// HMC algebra hooks (thin raw-pointer wrappers over math::su2). These keep `Hmc`
// field-agnostic: it loops directions and calls `G::sample_algebra_slab` /
// `kinetic_slab` / `expi_lmul_slab` per direction. The Wilson plaquette physics
// (Re Tr U_p, the staple force) is action-specific — see
// action/gauge/formula/wilson_su2.hpp (`wilson_kernels<SU2>`).
//
// Storage: full 2×2 complex matrix, 8 reals per link (see math::su2 for the
// component layout). N = 2. Matrix-link SU(2) is a `double`-only field type.

struct SU2 {
    using scalar_t                                 = double;
    static constexpr std::size_t n_real_components = 8;
    static constexpr std::size_t n_color           = 2;
    static constexpr std::string_view name         = "SU2";

    template <class T, class Rng>
    [[gnu::always_inline]] static inline void
    sample_algebra_slab(T* p_blk, Rng& rng, std::size_t n) noexcept {
        math::su2::sample_algebra_slab(p_blk, rng, n);
    }

    template <class T>
    [[gnu::always_inline]] static inline double kinetic_slab(T const* p_blk,
                                                             std::size_t n) noexcept {
        return math::su2::kinetic_slab(p_blk, n);
    }

    template <class T>
    [[gnu::always_inline]] static inline void
    expi_lmul_slab(T* u_blk, T const* p_blk, double dt, std::size_t n) noexcept {
        math::su2::expi_lmul_slab(u_blk, p_blk, dt, n);
    }
};

static_assert(GaugeGroup<SU2>);

}  // namespace reticolo::gauge_group
