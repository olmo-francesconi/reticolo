#pragma once

#include <reticolo/core/log/log.hpp>
#include <reticolo/core/rng/philox.hpp>
#include <reticolo/core/rng/rng.hpp>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <string_view>

namespace reticolo {

// Counter-based RNG satisfying the `Rng` concept — the CPU face of the shared
// Philox primitive (philox.hpp), so the same generator family runs on the CPU
// and on the GPU momentum sampler. A drop-in alongside FastRng; FastRng remains
// the serial default, PhiloxRng is the portable / parallel-reproducible option.
// The stream is the Philox bijection over an incrementing 64-bit counter; state
// is just (seed, counter), so a checkpoint is two integers.
class PhiloxRng {
public:
    using state_type = std::uint64_t;

    static constexpr std::string_view name = "PhiloxRng";

    explicit PhiloxRng(state_type seed    = 0xC0FFEEULL,
                       log::Mode announce = log::Mode::normal) noexcept
        : seed_{seed} {
        if (announce == log::Mode::normal) {
            log::info("rng", "PhiloxRng seed={:#x}", seed);
        }
    }

    // Independent stream k: same key, counter word-pair [2],[3] = k — every
    // stream draws from a disjoint 2^64 counter subspace (the incrementing
    // counter occupies words [0],[1]). Stream 0 is bit-identical to a plain
    // PhiloxRng with the same seed.
    [[nodiscard]] static PhiloxRng stream(state_type seed, std::uint64_t k) noexcept {
        PhiloxRng r{seed, log::Mode::silent};
        r.stream_ = k;
        return r;
    }

    // Full state as flat words for the multi-stream checkpoint layout:
    // [seed, stream, counter, idx, buf0, buf1, bit_cast(cached_normal),
    // has_cached_normal] — everything the generator owns, restored bit-exact
    // mid-buffer.
    static constexpr std::size_t n_state_words = 8;

    [[nodiscard]] std::array<state_type, n_state_words> state_words() const noexcept {
        return {seed_,
                stream_,
                counter_,
                static_cast<state_type>(idx_),
                buf_[0],
                buf_[1],
                std::bit_cast<state_type>(cached_normal_),
                has_cached_normal_ ? 1ULL : 0ULL};
    }

    [[nodiscard]] static PhiloxRng
    from_words(std::array<state_type, n_state_words> const& w) noexcept {
        PhiloxRng r{w[0], log::Mode::silent};
        r.stream_            = w[1];
        r.counter_           = w[2];
        r.idx_               = static_cast<std::size_t>(w[3]);
        r.buf_               = {w[4], w[5]};
        r.cached_normal_     = std::bit_cast<double>(w[6]);
        r.has_cached_normal_ = w[7] != 0;
        return r;
    }

    [[nodiscard]] state_type uniform_u64() noexcept {
        if (idx_ >= buf_.size()) {
            refill_();
        }
        return buf_[idx_++];
    }

    // 53-bit double in [0, 1).
    [[nodiscard]] double uniform() noexcept {
        return static_cast<double>(uniform_u64() >> 11U) * k_u53_scale;
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
            state_type const threshold = (0ULL - n) % n;
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

    // Standard normal via classical Box-Muller with a cached spare.
    [[nodiscard]] double normal() noexcept {
        if (has_cached_normal_) {
            has_cached_normal_ = false;
            return cached_normal_;
        }
        constexpr double k_two_pi = 2.0 * std::numbers::pi;
        double const u0           = std::max(uniform(), 1.0e-300);
        double const u1           = uniform();
        double const r            = std::sqrt(-2.0 * std::log(u0));
        double const theta        = k_two_pi * u1;
        cached_normal_            = r * std::sin(theta);
        has_cached_normal_        = true;
        return r * std::cos(theta);
    }

    // Batched standard-normal fill: writes `n` standard normals into `out`.
    // Classical Box-Muller in pairs — the same algorithm as `normal()`, so the
    // two paths share statistics. Drains the cached spare first, fills cos/sin
    // pairs, and tops off an odd tail via `normal()`. Lets PhiloxRng drive the
    // scalar/complex HMC momentum-sampling path (`updater::Hmc` calls normal_fill
    // for double and float fields), not just the matrix-group path.
    void normal_fill(double* out, std::size_t n) noexcept {
        constexpr double k_two_pi = 2.0 * std::numbers::pi;
        std::size_t i             = 0;
        if (has_cached_normal_ && i < n) {
            out[i++]           = cached_normal_;
            has_cached_normal_ = false;
        }
        for (; i + 1 < n; i += 2) {
            double const u0    = std::max(uniform(), 1.0e-300);
            double const u1    = uniform();
            double const r     = std::sqrt(-2.0 * std::log(u0));
            double const theta = k_two_pi * u1;
            out[i]             = r * std::cos(theta);
            out[i + 1]         = r * std::sin(theta);
        }
        if (i < n) {
            out[i] = normal();
        }
    }

private:
    void refill_() noexcept {
        Philox4x32::U32x2 const key{static_cast<std::uint32_t>(seed_),
                                    static_cast<std::uint32_t>(seed_ >> 32U)};
        Philox4x32::U32x4 const ctr{static_cast<std::uint32_t>(counter_),
                                    static_cast<std::uint32_t>(counter_ >> 32U),
                                    static_cast<std::uint32_t>(stream_),
                                    static_cast<std::uint32_t>(stream_ >> 32U)};
        Philox4x32::U32x4 const o = Philox4x32::bijection(ctr, key);
        buf_[0]                   = (static_cast<state_type>(o[1]) << 32U) | o[0];
        buf_[1]                   = (static_cast<state_type>(o[3]) << 32U) | o[2];
        idx_                      = 0;
        ++counter_;
    }

    state_type seed_;
    state_type stream_  = 0;
    state_type counter_ = 0;
    std::array<state_type, 2> buf_{};
    std::size_t idx_        = 2;  // forces a refill on first draw
    double cached_normal_   = 0.0;
    bool has_cached_normal_ = false;
};

static_assert(Rng<PhiloxRng>);

}  // namespace reticolo
