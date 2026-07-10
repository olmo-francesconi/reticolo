#pragma once

#include <concepts>
#include <cstdint>

// The RNG contract: the `Rng` concept every generator family satisfies
// (fast_rng / philox_rng / ranlxd_rng / mt19937_rng, and StreamSet by driver
// delegation), plus the SplitMix64 seed expander they share. Families live in
// their own `<name>_rng.hpp`; the counter-based Philox primitive shared with
// the CUDA backend is `philox.hpp`.

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

// SplitMix64 mix (the canonical seed expander; Steele, Lea & Flood 2014).
// Decorrelates per-stream seeds for engines without a jump function:
// stream k of a StreamSet seeds from splitmix64(seed + k·golden), the k-th
// element of the SplitMix64 sequence started at `seed`.
[[nodiscard]] constexpr std::uint64_t splitmix64(std::uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27U)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31U);
}

}  // namespace reticolo
