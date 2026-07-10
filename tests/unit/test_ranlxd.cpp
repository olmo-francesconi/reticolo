// RanlxdRng bit-compatibility suite. Pins the clean-room ranlxd generator to
// the reference ranlux v3.4 double stream (fixtures in ranlxd_fixtures.hpp are
// recorded program output, not source — see that file's provenance header):
// for a spread of (luxury level, seed) pairs the first 96 emitted doubles and
// the block at stream offset 1,000,000 must match bit-for-bit. Plus the
// portable state round-trip, stream(seed, k) distinctness, and the u64->31-bit
// seed mapping.

#include <reticolo/core/rng/ranlxd_rng.hpp>
#include <reticolo/core/rng/stream_set.hpp>

#include "ranlxd_fixtures.hpp"

#include <array>
#include <bit>
#include <cstdint>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using reticolo::RanlxdRng;
using reticolo::log::Mode;

namespace {
std::uint64_t bits(double d) { return std::bit_cast<std::uint64_t>(d); }

void check_stream(int level,
                  std::uint64_t seed,
                  std::array<std::uint64_t, 96> const& first96,
                  std::array<std::uint64_t, 24> const& after1e6) {
    RanlxdRng r{seed, level, Mode::silent};
    for (std::size_t i = 0; i < first96.size(); ++i) {
        REQUIRE(bits(r.uniform()) == first96[i]);
    }
    for (std::size_t i = first96.size(); i < 1'000'000; ++i) {
        (void)r.uniform();
    }
    for (std::size_t i = 0; i < after1e6.size(); ++i) {
        REQUIRE(bits(r.uniform()) == after1e6[i]);
    }
}
}  // namespace

TEST_CASE("RanlxdRng reproduces the reference ranlxd stream bit-for-bit", "[rng][ranlxd]") {
    using namespace ranlxd_fixtures;
    SECTION("level 2, seed 1") {
        check_stream(2, 1, level2_seed1_first96, level2_seed1_after1e6);
    }
    SECTION("level 2, seed 314159265") {
        check_stream(2, 314159265, level2_seed314159265_first96, level2_seed314159265_after1e6);
    }
    SECTION("level 1, seed 42") {
        check_stream(1, 42, level1_seed42_first96, level1_seed42_after1e6);
    }
    SECTION("level 1, seed 2147483646") {
        check_stream(
            1, 2147483646, level1_seed2147483646_first96, level1_seed2147483646_after1e6);
    }
}

TEST_CASE("RanlxdRng surfaces the 5-zero-low-bit ranlxd property", "[rng][ranlxd]") {
    // Each double is X / 2^48 with X a 48-bit integer, so the low 5 bits of the
    // 52-bit significand are always zero — a checkable RANLUX invariant.
    RanlxdRng r{12345, 2, Mode::silent};
    for (int i = 0; i < 4096; ++i) {
        REQUIRE((bits(r.uniform()) & 0x1FULL) == 0);
    }
}

TEST_CASE("RanlxdRng snapshot/continue equals restore-into-fresh", "[rng][ranlxd]") {
    RanlxdRng r{2718281, 2, Mode::silent};
    for (int i = 0; i < 5; ++i) {
        (void)r.normal();  // odd count -> live cached spare
    }
    for (int i = 0; i < 53; ++i) {
        (void)r.uniform();  // land mid-batch
    }
    auto const snap = r.state_words();

    std::vector<double> cont;
    for (int i = 0; i < 200; ++i) {
        cont.push_back(r.normal());
        cont.push_back(r.uniform());
    }

    RanlxdRng t = RanlxdRng::from_words(snap);
    for (int i = 0; i < 200; ++i) {
        REQUIRE(t.normal() == cont[static_cast<std::size_t>(2 * i)]);
        REQUIRE(t.uniform() == cont[static_cast<std::size_t>((2 * i) + 1)]);
    }
}

TEST_CASE("RanlxdRng::stream(seed, k) gives distinct sequences", "[rng][ranlxd]") {
    std::array<std::uint64_t, 8> firsts{};
    for (std::uint64_t k = 0; k < 8; ++k) {
        RanlxdRng s = RanlxdRng::stream(42, k);
        firsts[k]   = s.uniform_u64();
    }
    for (std::size_t i = 0; i < firsts.size(); ++i) {
        for (std::size_t j = i + 1; j < firsts.size(); ++j) {
            REQUIRE(firsts[i] != firsts[j]);
        }
    }
}

TEST_CASE("RanlxdRng u64-seed mapping is deterministic", "[rng][ranlxd]") {
    RanlxdRng a{123456789, 2, Mode::silent};
    RanlxdRng b{123456789, 2, Mode::silent};
    for (int i = 0; i < 64; ++i) {
        REQUIRE(a.uniform_u64() == b.uniform_u64());
    }
    // The mapping is the identity on the valid 31-bit range, so a raw seed and
    // its fold land on the same stream (seed 42 is already in [1, 2^31-1]).
    RanlxdRng c{42, 2, Mode::silent};
    RanlxdRng d = RanlxdRng::stream(42, 0);
    REQUIRE(c.uniform_u64() == d.uniform_u64());
}
