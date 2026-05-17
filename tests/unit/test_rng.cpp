#include <reticolo/core/rng.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::Rng;

static_assert(Rng<FastRng>, "FastRng must satisfy the Rng concept");

TEST_CASE("FastRng is deterministic for a given seed", "[rng]") {
    FastRng a{42};
    FastRng b{42};
    for (int i = 0; i < 1024; ++i) {
        REQUIRE(a.uniform_u64() == b.uniform_u64());
    }
}

TEST_CASE("FastRng reseed restarts the sequence", "[rng]") {
    FastRng r{1};
    std::array<std::uint64_t, 8> first{};
    for (auto& v : first) {
        v = r.uniform_u64();
    }
    r.reseed(1);
    for (auto v : first) {
        REQUIRE(r.uniform_u64() == v);
    }
}

TEST_CASE("Different seeds produce different first words", "[rng]") {
    FastRng a{0};
    FastRng b{1};
    REQUIRE(a.uniform_u64() != b.uniform_u64());
}

TEST_CASE("FastRng::uniform() stays in the unit interval", "[rng]") {
    FastRng r{7};
    for (int i = 0; i < 200'000; ++i) {
        double const u = r.uniform();
        REQUIRE(u >= 0.0);
        REQUIRE(u < 1.0);
    }
}

TEST_CASE("FastRng::uniform_int(n) stays in range and handles edges", "[rng]") {
    FastRng r{13};
    REQUIRE(r.uniform_int(0) == 0);
    REQUIRE(r.uniform_int(1) == 0);
    for (std::uint64_t n : {2ULL, 5ULL, 1024ULL, 1'000'000ULL}) {
        for (int i = 0; i < 10'000; ++i) {
            REQUIRE(r.uniform_int(n) < n);
        }
    }
}

TEST_CASE("FastRng::uniform_int distributes roughly uniformly", "[rng]") {
    constexpr int k_bins        = 16;
    constexpr int k_samples     = 320'000;
    constexpr double k_expected = static_cast<double>(k_samples) / k_bins;
    std::array<int, k_bins> counts{};

    FastRng r{2025};
    for (int i = 0; i < k_samples; ++i) {
        ++counts[r.uniform_int(k_bins)];
    }

    // Each bin should land within ~3% of the expected count.
    for (int c : counts) {
        REQUIRE(std::abs(static_cast<double>(c) - k_expected) < 0.03 * k_expected);
    }
}

TEST_CASE("FastRng::normal mean and stddev match standard normal", "[rng]") {
    constexpr int k_samples = 200'000;
    FastRng r{99};
    double sum = 0.0;
    double sq  = 0.0;
    for (int i = 0; i < k_samples; ++i) {
        double const x = r.normal();
        sum += x;
        sq += x * x;
    }
    double const mean   = sum / k_samples;
    double const var    = (sq / k_samples) - (mean * mean);
    double const stddev = std::sqrt(var);

    REQUIRE(std::abs(mean) < 0.02);
    REQUIRE(std::abs(stddev - 1.0) < 0.02);
}

TEST_CASE("Copying FastRng snapshots state and diverges after reseed", "[rng]") {
    FastRng a{55};
    (void)a.uniform_u64();
    (void)a.uniform_u64();

    FastRng b = a;  // snapshot at this point
    for (int i = 0; i < 4; ++i) {
        REQUIRE(a.uniform_u64() == b.uniform_u64());
    }
    // After 4 synchronised draws they remain in lockstep — both got the same
    // state and the algorithm is deterministic.

    // Re-seed one independently; sequences must diverge.
    b.reseed(99);
    bool diverged = false;
    for (int i = 0; i < 8; ++i) {
        if (a.uniform_u64() != b.uniform_u64()) {
            diverged = true;
            break;
        }
    }
    REQUIRE(diverged);
}
