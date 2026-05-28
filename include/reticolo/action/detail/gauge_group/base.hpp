#pragma once

#include <reticolo/math/vec_libm.hpp>

#include <concepts>
#include <cstddef>

namespace reticolo::gauge_group {

// Compile-time site-batch width for the batched SU(N) force kernels. The
// batched force vectorises a `for b in 0..k_gauge_batch` loop over packed
// sites, so to fill whole SIMD registers in single precision the batch must be
// a multiple of the float lane count. We take the wider of {float lane width,
// 8}: that keeps the well-tested 8 on NEON/SSE (4 lanes) and AVX2 (8 lanes) —
// where the neighbour-index gather is amortised over 8 sites — and grows to 16
// on AVX-512 so the float force uses full 512-bit registers (double then uses
// two registers per batch). The cap `RETICOLO_GAUGE_K_BATCH_MAX` lets wide
// ISAs that would spill the per-batch scratch (notably SU(3), 18 reals across
// many `[k_gauge_batch]` arrays) be dialed back at build time without touching
// the kernels — e.g. -DRETICOLO_GAUGE_K_BATCH_MAX=8.
#ifndef RETICOLO_GAUGE_K_BATCH_MAX
    #define RETICOLO_GAUGE_K_BATCH_MAX 16
#endif

[[nodiscard]] consteval std::size_t gauge_k_batch_() noexcept {
    std::size_t const w   = math::k_vec_width_f > 8 ? math::k_vec_width_f : std::size_t{8};
    std::size_t const cap = static_cast<std::size_t>(RETICOLO_GAUGE_K_BATCH_MAX);
    return w < cap ? w : cap;
}

inline constexpr std::size_t k_gauge_batch = gauge_k_batch_();

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
