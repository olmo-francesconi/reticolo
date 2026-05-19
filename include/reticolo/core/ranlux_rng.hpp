#pragma once

#include <reticolo/core/rng.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

namespace reticolo {

// =============================================================================
//  RanluxRng — thin wrapper around std::ranlux48 (subtract-with-carry with
//  Lüscher's discard).
//
//  Drop-in for FastRng: same constructor / reseed / uniform_u64 / uniform /
//  uniform_int / normal / normal_fill surface, same `state_type = uint64_t`,
//  same Box-Muller polar form for Gaussian sampling. Plug into any existing
//  templated call site (`Hmc<A, R, Integrator, Field>`, gauge group algebra
//  samplers, LLR replicas, …) by changing `FastRng` to `RanluxRng`.
//
//  Engine produces 48 random bits per call; uniform_u64 combines two calls
//  via `(r1 << 16) ^ r2` so all 96 source bits feed the 64-bit output (the
//  high 16 of the result are r1's top 16 bits unmixed; the rest is XORed
//  with r2). Quality of the discard schedule built into std::ranlux48 is
//  preserved — we're only repackaging bits, not introducing correlations.
// =============================================================================

class RanluxRng {
public:
    using state_type = std::uint64_t;

    explicit RanluxRng(state_type seed = 0xC0FFEEULL) noexcept : engine_{seed} {}

    void reseed(state_type seed) noexcept {
        engine_.seed(seed);
        has_cached_normal_ = false;
    }

    // engine_() returns std::ranlux48::result_type (= std::uint_fast64_t,
    // which is std::uint64_t on every supported target). Use `auto` and a
    // single result cast to keep GCC's -Wuseless-cast quiet on platforms
    // where result_type already == state_type.
    [[nodiscard]] state_type uniform_u64() noexcept {
        auto const r1 = engine_();
        auto const r2 = engine_();
        return (r1 << 16U) ^ r2;
    }

    [[nodiscard]] double uniform() noexcept {
        constexpr double k_scale = 1.0 / static_cast<double>(1ULL << 53U);
        return static_cast<double>(uniform_u64() >> 11U) * k_scale;
    }

    [[nodiscard]] state_type uniform_int(state_type n) noexcept {
        if (n <= 1) {
            return 0;
        }
#if defined(__SIZEOF_INT128__)
        auto m = static_cast<__uint128_t>(uniform_u64()) * static_cast<__uint128_t>(n);
        auto l = static_cast<state_type>(m);
        if (l < n) {
            state_type const threshold = (-n) % n;
            while (l < threshold) {
                m = static_cast<__uint128_t>(uniform_u64()) * static_cast<__uint128_t>(n);
                l = static_cast<state_type>(m);
            }
        }
        return static_cast<state_type>(m >> 64U);
#else
        return uniform_u64() % n;
#endif
    }

    [[nodiscard]] double normal() noexcept {
        if (has_cached_normal_) {
            has_cached_normal_ = false;
            return cached_normal_;
        }
        double u1 = 0.0;
        double u2 = 0.0;
        double s  = 0.0;
        do {
            u1 = (2.0 * uniform()) - 1.0;
            u2 = (2.0 * uniform()) - 1.0;
            s  = (u1 * u1) + (u2 * u2);
        } while (s >= 1.0 || s == 0.0);
        double const factor = std::sqrt(-2.0 * std::log(s) / s);
        cached_normal_      = u2 * factor;
        has_cached_normal_  = true;
        return u1 * factor;
    }

    void normal_fill(double* out, std::size_t n) noexcept {
        std::size_t i = 0;
        if (has_cached_normal_ && i < n) {
            out[i++]           = cached_normal_;
            has_cached_normal_ = false;
        }
        for (; i + 1 < n; i += 2) {
            double u1 = 0.0;
            double u2 = 0.0;
            double s  = 0.0;
            do {
                u1 = (2.0 * uniform()) - 1.0;
                u2 = (2.0 * uniform()) - 1.0;
                s  = (u1 * u1) + (u2 * u2);
            } while (s >= 1.0 || s == 0.0);
            double const factor = std::sqrt(-2.0 * std::log(s) / s);
            out[i]              = u1 * factor;
            out[i + 1]          = u2 * factor;
        }
        if (i < n) {
            out[i] = normal();
        }
    }

private:
    std::ranlux48 engine_;
    double cached_normal_   = 0.0;
    bool has_cached_normal_ = false;
};

static_assert(Rng<RanluxRng>);

}  // namespace reticolo
