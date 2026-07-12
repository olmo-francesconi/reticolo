#pragma once

#include <reticolo/core/exec/parallel.hpp>
#include <reticolo/core/log/log.hpp>
#include <reticolo/core/rng/rng.hpp>
#include <reticolo/math/vec_libm.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <string_view>
#include <vector>

namespace reticolo {

// xoshiro256++ (Blackman & Vigna 2018) with SplitMix64 seeding.
// Period 2^256 - 1, passes BigCrush, fast on every modern target.
class FastRng {
public:
    using state_type = std::uint64_t;

    static constexpr std::string_view name = "FastRng";

    explicit FastRng(state_type seed    = 0xC0FFEEULL,
                     log::Mode announce = log::Mode::normal) noexcept {
        reseed(seed);
        if (announce == log::Mode::normal) {
            log::info("rng", "FastRng  seed={:#x}", seed);
        }
    }

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

    // Advance 2^128 steps (Blackman & Vigna's published jump polynomial for
    // xoshiro256++): partitions the 2^256-1 period into provably disjoint
    // subsequences — StreamSet builds stream k as k+1 jumps from the seed.
    // Drops the cached normal (it belongs to the pre-jump position).
    void jump() noexcept {
        static constexpr std::array<state_type, 4> k_jump{0x180EC6D33CFD0ABAULL,
                                                          0xD5A61266F0C9392CULL,
                                                          0xA9582618E03FC9AAULL,
                                                          0x39ABDC4529B1661CULL};
        std::array<state_type, 4> t{};
        for (state_type const w : k_jump) {
            for (unsigned b = 0; b < 64; ++b) {
                if ((w & (1ULL << b)) != 0) {
                    for (std::size_t i = 0; i < 4; ++i) {
                        t[i] ^= s_[i];
                    }
                }
                (void)uniform_u64();
            }
        }
        s_                 = t;
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
    // Sleef's scalar `Sleef_sincos_u10` here; `normal_fill` uses the SIMD
    // sincos. Same algorithm, same seed → same stream to machine precision,
    // not bit-identical across ISAs (the SIMD reduction inside Sleef gives
    // different last-ULP results than the scalar path on x86 vs NEON).
    [[nodiscard]] double normal() noexcept {
        if (has_cached_normal_) {
            has_cached_normal_ = false;
            return cached_normal_;
        }
        constexpr double k_two_pi = 2.0 * std::numbers::pi;
        double const u1           = std::max(uniform(), 1.0e-300);
        double const u2           = uniform();
        double const r            = std::sqrt(-2.0 * std::log(u1));
        double const theta        = k_two_pi * u2;
#if defined(__CUDACC__)
        // nvcc cannot see <sleef.h> (guarded out of vec_libm.hpp); FastRng runs
        // host-side anyway, so the scalar std path is correct here. Same Sleef
        // isolation invariant as vec_libm.hpp / the CUDA backend.
        cached_normal_     = r * std::sin(theta);
        has_cached_normal_ = true;
        return r * std::cos(theta);
#else
        Sleef_double_2 const sc = Sleef_sincos_u10(theta);
        cached_normal_          = r * sc.x;  // sin
        has_cached_normal_      = true;
        return r * sc.y;  // cos
#endif
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
            thread_local std::vector<double> buf;
            double* const r_buf       = exec::thread_scratch(buf, 4 * n_pairs);
            double* const theta_buf   = r_buf + n_pairs;
            double* const sin_buf     = r_buf + (2 * n_pairs);
            double* const cos_buf     = r_buf + (3 * n_pairs);
            constexpr double k_two_pi = 2.0 * std::numbers::pi;
            for (std::size_t p = 0; p < n_pairs; ++p) {
                double const u1 = std::max(uniform(), 1.0e-300);
                double const u2 = uniform();
                r_buf[p]        = std::sqrt(-2.0 * std::log(u1));
                theta_buf[p]    = k_two_pi * u2;
            }
            math::sincos_batch(sin_buf, cos_buf, theta_buf, n_pairs);
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
    [[nodiscard]] double cached_normal() const noexcept { return cached_normal_; }
    [[nodiscard]] bool has_cached_normal() const noexcept { return has_cached_normal_; }

    // Rehydrate from a previously captured state — every word the generator
    // owns, restored bit-exact. Skips the constructor's `log::info` since
    // resume sites already announce themselves via io::Reader.
    [[nodiscard]] static FastRng
    from_state(std::array<state_type, 4> const& s, double cached, bool has_cached) noexcept {
        FastRng r{0, log::Mode::silent};
        r.s_                 = s;
        r.cached_normal_     = cached;
        r.has_cached_normal_ = has_cached;
        return r;
    }

    // Full state as flat words for the multi-stream checkpoint layout:
    // [s0..s3, bit_cast(cached_normal), has_cached_normal].
    static constexpr std::size_t n_state_words = 6;

    [[nodiscard]] std::array<state_type, n_state_words> state_words() const noexcept {
        return {s_[0],
                s_[1],
                s_[2],
                s_[3],
                std::bit_cast<state_type>(cached_normal_),
                has_cached_normal_ ? 1ULL : 0ULL};
    }

    [[nodiscard]] static FastRng
    from_words(std::array<state_type, n_state_words> const& w) noexcept {
        return from_state({w[0], w[1], w[2], w[3]}, std::bit_cast<double>(w[4]), w[5] != 0);
    }

private:
    std::array<state_type, 4> s_{};
    double cached_normal_   = 0.0;
    bool has_cached_normal_ = false;
};

static_assert(Rng<FastRng>);

}  // namespace reticolo
