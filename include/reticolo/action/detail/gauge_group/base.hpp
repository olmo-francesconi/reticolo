#pragma once

#include <concepts>
#include <cstddef>

namespace reticolo::gauge_group {

// GaugeGroup concept — type-level interface satisfied by every gauge group
// model (U(1), SU(2), SU(3), ...) that can serve as the template parameter
// to the generic `Wilson<G>` action.
//
// Members the concept requires:
//     scalar_t                    real type used for one stored component
//     n_real_components           # of doubles per stored link element
//                                   (1 for U(1) = the angle;
//                                    8 for SU(2) = 4 complex;
//                                    18 for SU(3) = 9 complex)
//     n_color                     N in SU(N) — used for the Wilson 1/N prefactor
//                                   and for the constant `N · n_plaq` in s_full
//
// Plaquette primitives (provided as static member templates on G; the
// concept doesn't require them syntactically — Wilson<G> instantiation
// will fail at compile time if a model is missing them, with a clear
// "no matching member function" error):
//
//     G::plaq_re_tr(mb, nb, s, s_pmu, s_pnu, stride) -> double
//        Re Tr U_p where U_p = U_mu(s)·U_nu(s+pmu)·U_mu(s+pnu)†·U_nu(s)†
//
//     G::plaq_force_accum(mb, nb, fmu_blk, fnu_blk,
//                         s, s_pmu, s_pnu, stride, beta_over_N)
//        Add the per-plaquette contribution to the 4 link forces it touches.
//
// Both primitives take pointers to the start of a direction's nc-component
// block (`mu_block_data(mu)` on MatrixLinkLattice) plus the per-direction
// stride (= nsites). Component k of the link at site s lives at
//     block_ptr[k * stride + s]
// For models with nc = 1 (U(1)), the stride argument is unused.

template <class G>
concept GaugeGroup = requires {
    typename G::scalar_t;
    requires std::convertible_to<decltype(G::n_real_components), std::size_t>;
    requires std::convertible_to<decltype(G::n_color), std::size_t>;
    requires G::n_real_components >= 1;
    requires G::n_color >= 1;
};

}  // namespace reticolo::gauge_group
