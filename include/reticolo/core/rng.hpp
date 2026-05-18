#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>

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

    // Standard normal via Box-Muller polar form; second sample cached. The cache
    // uses a plain (bool, double) pair — std::optional<double> here adds a branch
    // + indirection that is measurable in MC sweeps where normal() is the hottest
    // member.
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

    // Batched-fill: writes `n` standard normals into `out`. Equivalent to calling
    // normal() `n` times — same byte stream, same statistical properties — but the
    // hot loop is straight-line code with no per-call cache branch and no virtual
    // boundary, so the OoO engine sees more work to overlap. Used by HMC momentum
    // sampling where every site needs an independent normal.
    void normal_fill(double* out, std::size_t n) noexcept {
        std::size_t i = 0;
        if (has_cached_normal_ && i < n) {
            out[i++]           = cached_normal_;
            has_cached_normal_ = false;
        }
        // Emit pairs.
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
            out[i] = normal();  // updates the cache for the next call.
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
