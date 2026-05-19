#pragma once

#include <reticolo/math/vec_libm.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace reticolo {

// Minimal RNG interface used by the library. Implementations are stateful
// value types: copies do NOT alias the original state (each copy diverges
// independently from the moment it is made).
template <class R>
concept Rng = requires(R& r, std::uint64_t n) {
    { r.uniform_u64() } -> std::convertible_to<std::uint64_t>;   // [0, 2^64)
    { r.uniform() } -> std::convertible_to<double>;              // [0, 1)
    { r.normal() } -> std::convertible_to<double>;               // N(0, 1)
    { r.uniform_int(n) } -> std::convertible_to<std::uint64_t>;  // [0, n)
};

// xoshiro256++ (Blackman & Vigna 2018) with SplitMix64 seeding.
// Period 2^256 - 1, passes BigCrush, fast on every modern target.
class FastRng {
public:
    using state_type = std::uint64_t;

    explicit FastRng(state_type seed = 0xC0FFEEULL) noexcept { reseed(seed); }

    void reseed(state_type seed) noexcept {
        state_type sm = seed;
        for (auto& w : s_) {
            sm += 0x9E3779B97F4A7C15ULL;
            state_type z = sm;
            z            = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
            z            = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
            w            = z ^ (z >> 31U);
        }
        has_cached_normal_ = false;
    }

    [[nodiscard]] state_type uniform_u64() noexcept {
        state_type const result = std::rotl(s_[0] + s_[3], 23) + s_[0];
        state_type const t      = s_[1] << 17U;
        s_[2] ^= s_[0];
        s_[3] ^= s_[1];
        s_[1] ^= s_[2];
        s_[0] ^= s_[3];
        s_[2] ^= t;
        s_[3] = std::rotl(s_[3], 45);
        return result;
    }

    // 53-bit double in [0, 1).
    [[nodiscard]] double uniform() noexcept {
        constexpr double k_scale = 1.0 / static_cast<double>(1ULL << 53U);
        return static_cast<double>(uniform_u64() >> 11U) * k_scale;
    }

    // Lemire's debiased multiplication; returns 0 for n <= 1.
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
        // Portable fallback (slightly biased for huge n; fine for typical lattice volumes).
        return uniform_u64() % n;
#endif
    }

    // Standard normal via classical Box-Muller (sincos form, no rejection).
    // The polar form has data-dependent rejection that defeats vectorisation
    // in `normal_fill`; classical form has fixed cost (2 uniforms, 1 sqrt,
    // 1 log, 1 sincos per pair) and batches cleanly through Sleef. Uses
    // Sleef's scalar `Sleef_sincos_u10` so the single-call path is bit-
    // identical to the batched `normal_fill` path (which uses Sleef
    // `sincos_batch`) — the test suite enforces this agreement.
    [[nodiscard]] double normal() noexcept {
        if (has_cached_normal_) {
            has_cached_normal_ = false;
            return cached_normal_;
        }
        constexpr double k_two_pi = 6.283185307179586476925286766559;
        double const u1           = std::max(uniform(), 1.0e-300);
        double const u2           = uniform();
        double const r            = std::sqrt(-2.0 * std::log(u1));
        double const theta        = k_two_pi * u2;
        Sleef_double_2 const sc   = Sleef_sincos_u10(theta);
        cached_normal_            = r * sc.x;  // sin
        has_cached_normal_        = true;
        return r * sc.y;  // cos
    }

    // Batched-fill: writes `n` standard normals into `out`. Uses classical
    // Box-Muller with `sincos_batch` from vec_libm — one Sleef sincos call
    // per chunk of `k_vec_width_d` pairs. No rejection loop, no per-call
    // cache branch. Used by HMC momentum sampling where every site needs an
    // independent normal. The byte stream is not identical to repeated
    // `normal()` calls (different algorithm); statistical properties are.
    void normal_fill(double* out, std::size_t n) noexcept {
        std::size_t i = 0;
        if (has_cached_normal_ && i < n) {
            out[i++]           = cached_normal_;
            has_cached_normal_ = false;
        }

        std::size_t const remaining = n - i;
        std::size_t const n_pairs   = remaining / 2;

        if (n_pairs > 0) {
            thread_local std::vector<double> r_buf;
            thread_local std::vector<double> theta_buf;
            thread_local std::vector<double> sin_buf;
            thread_local std::vector<double> cos_buf;
            if (r_buf.size() < n_pairs) {
                r_buf.resize(n_pairs);
                theta_buf.resize(n_pairs);
                sin_buf.resize(n_pairs);
                cos_buf.resize(n_pairs);
            }
            constexpr double k_two_pi = 6.283185307179586476925286766559;
            for (std::size_t p = 0; p < n_pairs; ++p) {
                double const u1 = std::max(uniform(), 1.0e-300);
                double const u2 = uniform();
                r_buf[p]        = std::sqrt(-2.0 * std::log(u1));
                theta_buf[p]    = k_two_pi * u2;
            }
            math::sincos_batch(sin_buf.data(), cos_buf.data(), theta_buf.data(), n_pairs);
            for (std::size_t p = 0; p < n_pairs; ++p) {
                out[i + (2 * p)]     = r_buf[p] * cos_buf[p];
                out[i + (2 * p) + 1] = r_buf[p] * sin_buf[p];
            }
            i += 2 * n_pairs;
        }

        // Trailing odd element: generate a pair via normal() and cache the spare.
        if (i < n) {
            out[i] = normal();
        }
    }

    [[nodiscard]] std::array<state_type, 4> const& state() const noexcept { return s_; }

private:
    std::array<state_type, 4> s_{};
    double cached_normal_   = 0.0;
    bool has_cached_normal_ = false;
};

static_assert(Rng<FastRng>);

}  // namespace reticolo
