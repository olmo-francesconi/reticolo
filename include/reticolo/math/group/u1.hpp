#pragma once

#include <reticolo/core/rng/philox.hpp>
#include <reticolo/math/group/base.hpp>

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace reticolo::math::group {

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

    // (stride, count) form for layout uniformity with SU(N). U(1) has one real
    // component per link, so `stride` is irrelevant — only `count` links matter.
    template <class Rng>
    [[gnu::always_inline]] static inline void
    sample_algebra_slab(double* p_blk, Rng& rng, std::size_t stride, std::size_t count) noexcept {
        (void)stride;
        sample_algebra_slab(p_blk, rng, count);
    }

    // Counter-based (Philox) momentum sampler over links [base, base+cnt) of
    // direction `mu`: one N(0,1) draw per link, keyed by (key, mu, site) — a
    // pure function of the site index, so the draw worksplits and is
    // bit-identical for any thread count. `stride` unused (one real component
    // per link). Opt-in parallel replacement for `sample_algebra_slab`'s
    // serial FastRng fill.
    [[gnu::always_inline]] static inline void
    sample_algebra_philox_range(double* p_blk,
                                std::uint64_t key,
                                std::uint64_t mu,
                                std::size_t stride,
                                std::size_t base,
                                std::size_t cnt) noexcept {
        (void)stride;
        std::size_t const end = base + cnt;
        for (std::size_t s = base; s < end; ++s) {
            double n0 = 0.0;
            double n1 = 0.0;
            reticolo::philox_normal2(key, mu, s, n0, n1);
            p_blk[s] = n0;
        }
    }

    // Pure per-range kinetic worker: raw Σ p² (no ½) over [base, base+cnt).
    // The HMC kinetic reduce partitions the slab and folds these; the ½ is
    // applied once at the end.
    [[gnu::always_inline]] static inline double kinetic_range(double const* p_blk,
                                                              std::size_t stride,
                                                              std::size_t base,
                                                              std::size_t cnt) noexcept {
        (void)stride;
        double k              = 0.0;
        std::size_t const end = base + cnt;
        for (std::size_t s = base; s < end; ++s) {
            k += p_blk[s] * p_blk[s];
        }
        return k;
    }

    [[gnu::always_inline]] static inline double kinetic_slab(double const* p_blk,
                                                             std::size_t n) noexcept {
        return 0.5 * kinetic_range(p_blk, n, 0, n);
    }

    [[gnu::always_inline]] static inline double
    kinetic_slab(double const* p_blk, std::size_t stride, std::size_t count) noexcept {
        return 0.5 * kinetic_range(p_blk, stride, 0, count);
    }

    // Pure per-range drift worker: θ ← θ + dt·p over [base, base+cnt).
    // `stride` unused (one real component per link). The integrator op layer
    // partitions the slab and calls this per thread-chunk.
    [[gnu::always_inline]] static inline void expi_lmul_range(double* u_blk,
                                                              double const* p_blk,
                                                              double dt,
                                                              std::size_t stride,
                                                              std::size_t base,
                                                              std::size_t cnt) noexcept {
        (void)stride;
        std::size_t const end = base + cnt;
        for (std::size_t s = base; s < end; ++s) {
            u_blk[s] += dt * p_blk[s];
        }
    }

    // U(1) drift: U_new = exp(i·dt·p)·U_old reduces to θ_new = θ_old + dt·p.
    [[gnu::always_inline]] static inline void
    expi_lmul_slab(double* u_blk, double const* p_blk, double dt, std::size_t n) noexcept {
        expi_lmul_range(u_blk, p_blk, dt, n, 0, n);
    }

    [[gnu::always_inline]] static inline void expi_lmul_slab(double* u_blk,
                                                             double const* p_blk,
                                                             double dt,
                                                             std::size_t stride,
                                                             std::size_t count) noexcept {
        expi_lmul_range(u_blk, p_blk, dt, stride, 0, count);
    }
};

static_assert(GaugeGroup<U1>);

}  // namespace reticolo::math::group
