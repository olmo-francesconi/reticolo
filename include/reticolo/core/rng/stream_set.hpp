#pragma once

#include <reticolo/core/log/log.hpp>
#include <reticolo/core/exec/parallel.hpp>
#include <reticolo/core/rng/rng.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

// A pool of independent streams over one Rng family: one dedicated DRIVER
// stream (the serial draws — Metropolis accept, LLR exchange, hot starts) + n
// SITE streams. The pool holds NO geometry: which sites a stream serves is the
// owner's contract — updater::Hmc sizes the pool from the canonical field
// partition (exec::partition) and binds site stream i to partition item i via
// exec::field_visit_indexed, one-to-one, checked every fill. Serial draws
// never touch a site stream, so per-slab draw sequences stay shape-determined
// regardless of how many accept/exchange draws a run makes. A checkpoint is
// the state of every stream (state_words / restore_state_words).
//
// Stream construction is the per-family independence mechanism; a family
// provides exactly one of:
//   * jump()          — advance one whole subsequence (xoshiro256++'s 2^128
//                       jump polynomial): stream k = k+1 jumps from the seed.
//                       Provably disjoint subsequences.
//   * stream(seed, k) — open seed/counter subspace k (Philox counter words;
//                       SplitMix-decorrelated seeds for the std engines).

namespace reticolo {

template <class R>
concept JumpStream = Rng<R> && requires(R& r, R::state_type seed) {
    r.jump();
    R{seed, log::Mode::silent};
};

template <class R>
concept KeyedStream = Rng<R> && requires(R::state_type seed, std::uint64_t k) {
    { R::stream(seed, k) } -> std::same_as<R>;
};

template <class R>
    requires JumpStream<R> || KeyedStream<R>
class StreamSet {
public:
    using rng_type   = R;
    using state_type = R::state_type;

    StreamSet(state_type seed, std::size_t n_site_streams, log::Mode announce = log::Mode::normal) {
        std::size_t const ns = std::max<std::size_t>(n_site_streams, 1);
        streams_.reserve(ns + 1);
        if constexpr (JumpStream<R>) {
            R r{seed, log::Mode::silent};
            for (std::size_t k = 0; k <= ns; ++k) {
                streams_.push_back(Slot_{r});  // copies diverge independently (Rng contract)
                r.jump();
            }
        } else {
            for (std::size_t k = 0; k <= ns; ++k) {
                streams_.push_back(Slot_{R::stream(seed, k)});
            }
        }
        if (announce == log::Mode::normal) {
            log::info(
                "rng", "StreamSet<{}>  1 driver + {} site streams  seed={:#x}", R::name, ns, seed);
        }
    }

    // Stream 0 is the driver; site stream k is streams_[k + 1].
    [[nodiscard]] std::size_t n_streams() const noexcept { return streams_.size() - 1; }
    [[nodiscard]] R& driver() noexcept { return streams_.front().r; }
    [[nodiscard]] R const& driver() const noexcept { return streams_.front().r; }
    [[nodiscard]] R& site_stream(std::size_t k) noexcept { return streams_[k + 1].r; }
    [[nodiscard]] R const& site_stream(std::size_t k) const noexcept { return streams_[k + 1].r; }

    // Rng concept — serial draws delegate to the driver stream, so a StreamSet
    // serves every templated call site (Hmc accept, orch::llr::try_exchange, …).
    [[nodiscard]] std::uint64_t uniform_u64() noexcept { return driver().uniform_u64(); }
    [[nodiscard]] double uniform() noexcept { return driver().uniform(); }
    [[nodiscard]] double normal() noexcept { return driver().normal(); }
    [[nodiscard]] std::uint64_t uniform_int(std::uint64_t n) noexcept {
        return driver().uniform_int(n);
    }
    void normal_fill(double* out, std::size_t n) noexcept { driver().normal_fill(out, n); }

    // Flat state of every stream, driver first — (n_streams+1)·words_per_stream
    // words, the payload of the multi-stream /rng checkpoint dataset.
    static constexpr std::size_t words_per_stream = R::n_state_words;

    [[nodiscard]] std::vector<std::uint64_t> state_words() const {
        std::vector<std::uint64_t> w;
        w.reserve(streams_.size() * words_per_stream);
        for (Slot_ const& s : streams_) {
            auto const sw = s.r.state_words();
            w.insert(w.end(), sw.begin(), sw.end());
        }
        return w;
    }

    void restore_state_words(std::vector<std::uint64_t> const& w) {
        if (w.size() != streams_.size() * words_per_stream) {
            throw std::invalid_argument{"StreamSet::restore_state_words: word count mismatch"};
        }
        std::array<std::uint64_t, words_per_stream> a{};
        for (std::size_t k = 0; k < streams_.size(); ++k) {
            std::copy_n(w.begin() + static_cast<std::ptrdiff_t>(k * words_per_stream),
                        words_per_stream,
                        a.begin());
            streams_[k].r = R::from_words(a);
        }
    }

private:
    // Each stream sits alone on its own destructive-interference region.
    // Engine state is WRITTEN on every draw; packed sub-line generators
    // (PhiloxRng 64 B, ranlux48 ~136 B vs 128 B lines on Apple Silicon)
    // otherwise ping-pong cache lines between the fill threads — measured
    // 1.7-2.1× on a 4-thread uniform fill. Padding changes no bits: same
    // streams, same draws, same order.
    static constexpr std::size_t k_stream_align = exec::k_cache_line_bytes;
    struct alignas(k_stream_align) Slot_ {
        R r;
    };
    std::vector<Slot_> streams_;  // [0] = driver, [1 ..] = site streams
};

}  // namespace reticolo
