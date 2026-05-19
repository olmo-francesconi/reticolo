#pragma once

#include <reticolo/action/detail/gauge_group/base.hpp>
#include <reticolo/math/su2_ops.hpp>

#include <cstddef>

namespace reticolo::gauge_group {

// =============================================================================
//  SU(2) gauge group model. Storage: full 2×2 complex matrix, 8 reals per
//  link element (see math::su2 for the component layout). N = 2, so the
//  Wilson prefactor is β/2 and the per-link constant in s_full is 2·n_plaq.
//
//  This M4 commit ships the GaugeGroup-concept-mandatory members plus the
//  `plaq_re_tr` primitive that `Wilson<G>::s_full` and `s_local` need —
//  enough to *compute* the Wilson action on an SU(2) configuration without
//  yet having an HMC force kernel. `plaq_force_accum` (and therefore
//  `Wilson<SU2>::compute_force`) arrives in M5 once the staple-scatter
//  pattern is exercised end-to-end.
// =============================================================================

struct SU2 {
    using scalar_t                                 = double;
    static constexpr std::size_t n_real_components = 8;
    static constexpr std::size_t n_color           = 2;

    // Re Tr (U_mu(s) · U_nu(s+pmu) · U_mu(s+pnu)† · U_nu(s)†)
    // = Re Tr (AB · DC†) where AB = U_mu(s)·U_nu(s+pmu) and DC = U_nu(s)·U_mu(s+pnu).
    // Using Re Tr (X · Y†) = sum_{ij} [Re X_{ij}·Re Y_{ij} + Im X_{ij}·Im Y_{ij}]
    // — the inner product of the two matrices viewed as 8-real vectors.
    template <class T>
    [[gnu::always_inline]] static inline double
    plaq_re_tr(T const* mb,
               T const* nb,
               std::size_t s,
               std::size_t s_pmu,
               std::size_t s_pnu,
               std::size_t stride) noexcept {
        double a_mat[8];
        double b_mat[8];
        double c_mat[8];
        double d_mat[8];
        for (std::size_t k = 0; k < 8; ++k) {
            std::size_t const off = k * stride;
            a_mat[k]              = static_cast<double>(mb[off + s]);
            b_mat[k]              = static_cast<double>(nb[off + s_pmu]);
            c_mat[k]              = static_cast<double>(mb[off + s_pnu]);
            d_mat[k]              = static_cast<double>(nb[off + s]);
        }
        double ab[8];
        double dc[8];
        math::su2::mul_2x2(ab, a_mat, b_mat);
        math::su2::mul_2x2(dc, d_mat, c_mat);
        double re_tr = 0.0;
        for (std::size_t k = 0; k < 8; ++k) {
            re_tr += ab[k] * dc[k];
        }
        return re_tr;
    }
};

static_assert(GaugeGroup<SU2>);

}  // namespace reticolo::gauge_group
