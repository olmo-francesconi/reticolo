#pragma once

#include <reticolo/math/vec_libm.hpp>

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace reticolo::gauge_group {

// Compile-time site-batch width for the batched SU(N) kernels. The batched
// kernels vectorise a `for b in 0..k_gauge_batch<T>` loop over packed sites,
// so the batch must be a multiple of the lane count of the field's scalar
// type T. We take the wider of {T's lane width, 8}: 8 on NEON/SSE/AVX2 for
// both precisions (the neighbour-index gather is amortised over 8 sites);
// on AVX-512 the float batch would grow to 16 — but compilers default to
// 256-bit vectors even with -mavx512f (prefer-vector-width=256), where a
// 16-wide batch only doubles the per-batch scratch (notably SU(3), 18 reals
// across many `[k_batch]` arrays) and spills. The cap therefore defaults to
// 8; building for full 512-bit registers is the opt-in combo
// -mprefer-vector-width=512 -DRETICOLO_GAUGE_K_BATCH_MAX=16 (double stays at
// 8 = one zmm per row; only float grows to 16).
#ifndef RETICOLO_GAUGE_K_BATCH_MAX
    #define RETICOLO_GAUGE_K_BATCH_MAX 8
#endif

template <class T>
[[nodiscard]] consteval std::size_t gauge_k_batch_() noexcept {
    std::size_t const lanes = std::is_same_v<T, float> ? math::k_vec_width_f : math::k_vec_width_d;
    std::size_t const w     = lanes > 8 ? lanes : std::size_t{8};
    std::size_t const cap   = static_cast<std::size_t>(RETICOLO_GAUGE_K_BATCH_MAX);
    return w < cap ? w : cap;
}

template <class T>
inline constexpr std::size_t k_gauge_batch = gauge_k_batch_<T>();

// Load the (Re, Im) component slabs of one direction's links for a batch of
// B sites into `[E][B]` AoSoA scratch (E = N² complex entries; component k's
// Re at blk[(2k)·ns + s], Im at blk[(2k+1)·ns + s]).
//
// The gather form takes a per-lane site-index array. Neighbour batches are
// contiguous (idx[b] == idx[0] + b) for every bulk batch — including batches
// that wrap uniformly in a direction ≥ 1 — so the loads collapse to stride-1
// vector loads there; only batches straddling a wrap (one per row for μ = 0,
// row-crossing batches when L0 % B ≠ 0) pay the scalar gather.

template <std::size_t E, std::size_t B, class T>
[[gnu::always_inline]] inline void load_links_batched(
    T (&m_re)[E][B], T (&m_im)[E][B], T const* blk, std::size_t ns, std::size_t s0) noexcept {
    for (std::size_t k = 0; k < E; ++k) {
        std::size_t const off_re = (2 * k) * ns;
        std::size_t const off_im = off_re + ns;
        for (std::size_t b = 0; b < B; ++b) {
            m_re[k][b] = blk[off_re + s0 + b];
            m_im[k][b] = blk[off_im + s0 + b];
        }
    }
}

template <std::size_t E, std::size_t B, class T>
[[gnu::always_inline]] inline void load_links_batched(T (&m_re)[E][B],
                                                      T (&m_im)[E][B],
                                                      T const* blk,
                                                      std::size_t ns,
                                                      std::size_t const (&idx)[B]) noexcept {
    bool contig = true;
    for (std::size_t b = 1; b < B; ++b) {
        contig &= (idx[b] == idx[0] + b);
    }
    if (contig) [[likely]] {
        load_links_batched(m_re, m_im, blk, ns, idx[0]);
        return;
    }
    for (std::size_t k = 0; k < E; ++k) {
        std::size_t const off_re = (2 * k) * ns;
        std::size_t const off_im = off_re + ns;
        for (std::size_t b = 0; b < B; ++b) {
            m_re[k][b] = blk[off_re + idx[b]];
            m_im[k][b] = blk[off_im + idx[b]];
        }
    }
}

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
