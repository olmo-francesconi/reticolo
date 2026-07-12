#pragma once

#include <reticolo/core/log/log.hpp>
#include <reticolo/core/rng/rng.hpp>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace reticolo {

// RanlxdRng — clean-room reimplementation of Martin Lüscher's RANLUX
// double-precision generator (`ranlxd`, v3.4), bit-compatible with the
// reference: for a given (luxury level, 31-bit seed) `uniform()` reproduces
// the reference `ranlxd()` double stream exactly.
//
// Core (Lüscher 1994 + v3.4 algorithm notes): a subtract-with-carry recursion
// reformulated at double-word precision — base B = 2^48, long lag R = 12,
// short lag S = 5 — run as 4 independent copies in lockstep. Each batch runs
// the recursion for P steps (P = 202 at level 1, P = 397 at level 2) and
// surfaces only the trailing 12 slots of each copy (the decimation that gives
// RANLUX its decorrelation "luxury"). Integer arithmetic throughout: bit-exact
// and portable, no SIMD.
//
// Drop-in for the other Rng families: same state_type / uniform_u64 / uniform /
// uniform_int / normal / normal_fill surface. Replaces the retired std::ranlux48
// wrapper; unlike it, `stream(seed, k)` gives PROVABLY distinct trajectories
// (Lüscher's seeding guarantee, see stream()), and the checkpoint layout is
// fully portable flat words (no std-engine textual state).

class RanlxdRng {
    static constexpr int k_copies                = 4;   // parallel SWC copies
    static constexpr int k_ring                  = 12;  // long lag R (slots per copy)
    static constexpr int k_short_lag             = 5;   // short lag S
    static constexpr int k_ring_words            = k_copies * k_ring;  // 48
    static constexpr int k_batch                 = k_copies * k_ring;  // 48 surfaced per batch
    static constexpr int k_p_level1              = 202;
    static constexpr int k_p_level2              = 397;
    static constexpr std::uint64_t k_base        = 1ULL << 48U;  // B
    static constexpr double k_u48_scale          = 1.0 / static_cast<double>(1ULL << 48U);
    static constexpr std::uint64_t k_lfsr_period = (1ULL << 31U) - 1U;  // 2^31 - 1

public:
    using state_type = std::uint64_t;

    static constexpr std::string_view name = "RanlxdRng";

    // level 2 (P=397) is the default. `seed` is any u64: it is folded into the
    // reference's valid 31-bit seed range [1, 2^31-1] by to_seed31_(), which is
    // the identity on values already in that range (so seed 1, 42, ... match the
    // reference seeded with the same integer).
    explicit RanlxdRng(state_type seed    = 0xC0FFEEULL,
                       int level          = 2,
                       log::Mode announce = log::Mode::normal) noexcept
        : level_{level == 1 ? 1 : 2}, p_{level == 1 ? k_p_level1 : k_p_level2} {
        seed_ranlxd_(to_seed31_(seed));
        if (announce == log::Mode::normal) {
            log::info("rng", "RanlxdRng seed={:#x} level={}", seed, level_);
        }
    }

    void reseed(state_type seed) noexcept { seed_ranlxd_(to_seed31_(seed)); }

    // Independent stream k: sequential 31-bit seeds. base = to_seed31_(seed);
    // stream k is seeded with ((base - 1 + k) mod (2^31 - 1)) + 1. Distinct k
    // (mod 2^31-1) give distinct 31-bit seeds, and Lüscher's seeding scheme
    // (the primitive-trinomial LFSR of §6) guarantees distinct seeds produce
    // distinct, non-overlapping trajectories — a real disjointness guarantee,
    // not the decorrelated-seed heuristic the std wrappers relied on. Level 2.
    [[nodiscard]] static RanlxdRng stream(state_type seed, std::uint64_t k) noexcept {
        std::uint64_t const base = to_seed31_(seed);
        std::uint64_t const s31  = (((base - 1) + (k % k_lfsr_period)) % k_lfsr_period) + 1;
        return RanlxdRng{s31, 2, log::Mode::silent};
    }

    // Native ranlxd double: X / 2^48 with X the next 48-bit surfaced value.
    // Exact in IEEE-754 (48 significant bits fit in the 53-bit significand);
    // the low 5 significand bits are always zero, a checkable RANLUX property.
    [[nodiscard]] double uniform() noexcept {
        return static_cast<double>(next_raw_()) * k_u48_scale;
    }

    // Two consecutive 48-bit draws combined so all 64 result bits carry entropy:
    // (x0 << 16) ^ x1 — x0 supplies bits [16,64) (its top 16 unshared), x1 bits
    // [0,48). Deterministic; consumes exactly two native draws.
    [[nodiscard]] state_type uniform_u64() noexcept {
        state_type const x0 = next_raw_();
        state_type const x1 = next_raw_();
        return (x0 << 16U) ^ x1;
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
        return uniform_u64() % n;
#endif
    }

    // Standard normal via Box-Muller polar form with a cached spare.
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

    // Fully-portable flat-u64 checkpoint. Captures the complete generator state:
    // the 4x12 ring words, the 4 per-copy carries, the shared cursor, the raw
    // 48-value output buffer + its read position (so a mid-batch resume is
    // bit-exact), the luxury level, and the cached normal via bit_cast + flag.
    // Layout: [ring(48), carry(4), cursor, buf(48), buf_pos, level,
    // bit_cast(cached_normal), has_cached_normal].
    static constexpr std::size_t n_state_words =
        k_ring_words + k_copies + 1 + k_batch + 1 + 1 + 1 + 1;  // = 105

    [[nodiscard]] std::array<state_type, n_state_words> state_words() const noexcept {
        std::array<state_type, n_state_words> w{};
        std::size_t o = 0;
        for (state_type const v : ring_) {
            w[o++] = v;
        }
        for (state_type const v : carry_) {
            w[o++] = v;
        }
        w[o++] = static_cast<state_type>(cursor_);
        for (state_type const v : buf_) {
            w[o++] = v;
        }
        w[o++] = static_cast<state_type>(buf_pos_);
        w[o++] = static_cast<state_type>(level_);
        w[o++] = std::bit_cast<state_type>(cached_normal_);
        w[o++] = has_cached_normal_ ? 1ULL : 0ULL;
        return w;
    }

    [[nodiscard]] static RanlxdRng from_words(std::array<state_type, n_state_words> const& w) {
        RanlxdRng r{1, 2, log::Mode::silent};
        std::size_t o = 0;
        for (state_type& v : r.ring_) {
            v = w[o++];
        }
        for (state_type& v : r.carry_) {
            v = w[o++];
        }
        r.cursor_ = static_cast<int>(w[o++]);
        for (state_type& v : r.buf_) {
            v = w[o++];
        }
        r.buf_pos_           = static_cast<int>(w[o++]);
        r.level_             = static_cast<int>(w[o++]);
        r.p_                 = (r.level_ == 1) ? k_p_level1 : k_p_level2;
        r.cached_normal_     = std::bit_cast<double>(w[o++]);
        r.has_cached_normal_ = w[o] != 0;
        return r;
    }

private:
    // Fold any u64 into [1, 2^31 - 1]; identity on values already in range.
    [[nodiscard]] static std::uint64_t to_seed31_(std::uint64_t seed) noexcept {
        return (((seed - 1) % k_lfsr_period) % k_lfsr_period) + 1;
    }

    // §6 seeding: 31-bit LFSR b_n = b_{n-13} XOR b_{n-31} (b_0..b_30 = seed bits,
    // LSB first) → 2304 bits consumed MSB-first into 4x24 24-bit words → per-copy
    // complement (ranlxd rule: complement unless k mod 4 == i) → pairwise-pack
    // into the 12 ring slots (word 2j low 24 bits, word 2j+1 high 24 bits).
    void seed_ranlxd_(std::uint64_t s31) noexcept {
        std::array<std::uint8_t, 31> reg{};
        for (int n = 0; n < 31; ++n) {
            reg[static_cast<std::size_t>(n)] = static_cast<std::uint8_t>((s31 >> n) & 1U);
        }
        int p         = 0;
        auto next_bit = [&reg, &p]() noexcept -> std::uint32_t {
            std::uint8_t const b = reg[static_cast<std::size_t>(p)];
            reg[static_cast<std::size_t>(p)] ^= reg[static_cast<std::size_t>((p + 18) % 31)];
            p = (p + 1) % 31;
            return b;
        };

        for (int i = 0; i < k_copies; ++i) {
            for (int k = 0; k < 24; ++k) {
                std::uint32_t word = 0;
                for (int bit = 0; bit < 24; ++bit) {
                    word = (word << 1U) | next_bit();
                }
                if ((k % 4) != i) {
                    word = 0xFFFFFFU - word;  // 24-bit one's complement
                }
                std::size_t const slot =
                    (static_cast<std::size_t>(i) * k_ring) + static_cast<std::size_t>(k / 2);
                if ((k % 2) == 0) {
                    ring_[slot] = word;
                } else {
                    ring_[slot] |= static_cast<std::uint64_t>(word) << 24U;
                }
            }
        }
        for (std::uint64_t& c : carry_) {
            c = 0;
        }
        cursor_            = 0;
        buf_pos_           = k_batch;  // force a decimation pass on the first draw
        has_cached_normal_ = false;
    }

    // §3 decimation: advance the shared recursion p_ steps, then surface the
    // trailing 12 slots of each copy in cursor-rotated order (copies 0..3 fixed).
    void generate_() noexcept {
        for (int step = 0; step < p_; ++step) {
            int const short_slot = (cursor_ + (k_ring - k_short_lag)) % k_ring;
            for (int c = 0; c < k_copies; ++c) {
                std::size_t const base = static_cast<std::size_t>(c) * k_ring;
                std::uint64_t const s  = ring_[base + static_cast<std::size_t>(short_slot)];
                std::uint64_t const l  = ring_[base + static_cast<std::size_t>(cursor_)];
                auto d = static_cast<std::int64_t>(s) - static_cast<std::int64_t>(l) -
                         static_cast<std::int64_t>(carry_[static_cast<std::size_t>(c)]);
                if (d < 0) {
                    d += static_cast<std::int64_t>(k_base);
                    carry_[static_cast<std::size_t>(c)] = 1;
                } else {
                    carry_[static_cast<std::size_t>(c)] = 0;
                }
                ring_[base + static_cast<std::size_t>(cursor_)] = static_cast<std::uint64_t>(d);
            }
            cursor_ = (cursor_ + 1) % k_ring;
        }
        int idx = 0;
        for (int j = 0; j < k_ring; ++j) {
            int const slot = (cursor_ + j) % k_ring;
            for (int c = 0; c < k_copies; ++c) {
                buf_[static_cast<std::size_t>(idx++)] =
                    ring_[(static_cast<std::size_t>(c) * k_ring) + static_cast<std::size_t>(slot)];
            }
        }
        buf_pos_ = 0;
    }

    [[nodiscard]] std::uint64_t next_raw_() noexcept {
        if (buf_pos_ >= k_batch) {
            generate_();
        }
        return buf_[static_cast<std::size_t>(buf_pos_++)];
    }

    std::array<std::uint64_t, k_ring_words> ring_{};  // [copy*12 + slot], 0 <= X < 2^48
    std::array<std::uint64_t, k_copies> carry_{};     // per-copy carry bit (0/1)
    std::array<std::uint64_t, k_batch> buf_{};        // raw surfaced 48-bit values
    int cursor_             = 0;
    int buf_pos_            = k_batch;
    int level_              = 2;
    int p_                  = k_p_level2;
    double cached_normal_   = 0.0;
    bool has_cached_normal_ = false;
};

static_assert(Rng<RanlxdRng>);

}  // namespace reticolo
