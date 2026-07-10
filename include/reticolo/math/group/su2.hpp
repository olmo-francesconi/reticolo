#pragma once

#include <reticolo/math/group/base.hpp>
#include <reticolo/math/su2_ops.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace reticolo::math::group {

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

    // Padded-stride form: `count` links at component stride `stride` (≥ count).
    template <class T, class Rng>
    [[gnu::always_inline]] static inline void
    sample_algebra_slab(T* p_blk, Rng& rng, std::size_t stride, std::size_t count) noexcept {
        math::su2::sample_algebra_slab(p_blk, rng, stride, count);
    }

    template <class T>
    [[gnu::always_inline]] static inline double kinetic_slab(T const* p_blk,
                                                             std::size_t n) noexcept {
        return math::su2::kinetic_slab(p_blk, n);
    }

    template <class T>
    [[gnu::always_inline]] static inline double
    kinetic_slab(T const* p_blk, std::size_t stride, std::size_t count) noexcept {
        return math::su2::kinetic_slab(p_blk, stride, count);
    }

    // Pure per-range kinetic worker: raw Σ (h₁²+h₂²+h₃²) over [base, base+cnt).
    // The HMC kinetic reduce partitions the slab and folds these.
    template <class T>
    [[gnu::always_inline]] static inline double
    kinetic_range(T const* p_blk, std::size_t stride, std::size_t base, std::size_t cnt) noexcept {
        return math::su2::kinetic_range(p_blk, stride, base, cnt);
    }

    // Counter-based (Philox) momentum sampler over [base, base+cnt) of direction
    // `mu`. Opt-in parallel replacement for sample_algebra_slab's serial FastRng
    // fill; the HMC layer partitions the slab and draws `key` once per trajectory.
    template <class T>
    [[gnu::always_inline]] static inline void
    sample_algebra_philox_range(T* p_blk,
                                std::uint64_t key,
                                std::uint64_t mu,
                                std::size_t stride,
                                std::size_t base,
                                std::size_t cnt) noexcept {
        math::su2::sample_algebra_philox_range(p_blk, key, mu, stride, base, cnt);
    }

    template <class T>
    [[gnu::always_inline]] static inline void
    expi_lmul_slab(T* u_blk, T const* p_blk, double dt, std::size_t n) noexcept {
        math::su2::expi_lmul_slab(u_blk, p_blk, dt, n);
    }

    template <class T>
    [[gnu::always_inline]] static inline void expi_lmul_slab(
        T* u_blk, T const* p_blk, double dt, std::size_t stride, std::size_t count) noexcept {
        math::su2::expi_lmul_slab(u_blk, p_blk, dt, stride, count);
    }

    // Pure per-range drift worker (U ← exp(dt·P)·U over [base, base+cnt) with
    // component stride `stride`). The integrator op layer partitions the slab
    // and calls this per thread-chunk; the group/math stay threading-free.
    template <class T>
    [[gnu::always_inline]] static inline void expi_lmul_range(T* u_blk,
                                                              T const* p_blk,
                                                              double dt,
                                                              std::size_t stride,
                                                              std::size_t base,
                                                              std::size_t cnt) noexcept {
        math::su2::expi_lmul_range(u_blk, p_blk, dt, stride, base, cnt);
    }
};

static_assert(GaugeGroup<SU2>);

}  // namespace reticolo::math::group
