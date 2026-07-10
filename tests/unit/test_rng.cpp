#include <reticolo/core/rng/fast_rng.hpp>
#include <reticolo/core/rng/mt19937_rng.hpp>
#include <reticolo/core/rng/ranlxd_rng.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::Mt19937Rng;
using reticolo::RanlxdRng;
using reticolo::Rng;

static_assert(Rng<FastRng>, "FastRng must satisfy the Rng concept");
static_assert(Rng<RanlxdRng>, "RanlxdRng must satisfy the Rng concept");
static_assert(Rng<Mt19937Rng>, "Mt19937Rng must satisfy the Rng concept");

// Library-wide RNG correctness suite. The same checks apply to every RNG
// that satisfies the Rng concept; TEMPLATE_TEST_CASE expands one set of
// asserts per (test, RNG) pair so a future fourth RNG just needs adding
// to the type list. Statistical thresholds are loose enough to clear any
// well-mixed engine.

TEMPLATE_TEST_CASE(
    "Rng is deterministic for a given seed", "[rng]", FastRng, RanlxdRng, Mt19937Rng) {
    TestType a{42};
    TestType b{42};
    for (int i = 0; i < 1024; ++i) {
        REQUIRE(a.uniform_u64() == b.uniform_u64());
    }
}

TEMPLATE_TEST_CASE("Rng reseed restarts the sequence", "[rng]", FastRng, RanlxdRng, Mt19937Rng) {
    TestType r{1};
    std::array<std::uint64_t, 8> first{};
    for (auto& v : first) {
        v = r.uniform_u64();
    }
    r.reseed(1);
    for (auto v : first) {
        REQUIRE(r.uniform_u64() == v);
    }
}

TEMPLATE_TEST_CASE(
    "Different seeds produce different first words", "[rng]", FastRng, RanlxdRng, Mt19937Rng) {
    TestType a{0};
    TestType b{1};
    REQUIRE(a.uniform_u64() != b.uniform_u64());
}

TEMPLATE_TEST_CASE(
    "Rng::uniform() stays in the unit interval", "[rng]", FastRng, RanlxdRng, Mt19937Rng) {
    TestType r{7};
    for (int i = 0; i < 200'000; ++i) {
        double const u = r.uniform();
        REQUIRE(u >= 0.0);
        REQUIRE(u < 1.0);
    }
}

TEMPLATE_TEST_CASE("Rng::uniform_int(n) stays in range and handles edges",
                   "[rng]",
                   FastRng,
                   RanlxdRng,
                   Mt19937Rng) {
    TestType r{13};
    REQUIRE(r.uniform_int(0) == 0);
    REQUIRE(r.uniform_int(1) == 0);
    for (std::uint64_t n : {2ULL, 5ULL, 1024ULL, 1'000'000ULL}) {
        for (int i = 0; i < 10'000; ++i) {
            REQUIRE(r.uniform_int(n) < n);
        }
    }
}

TEMPLATE_TEST_CASE(
    "Rng::uniform_int distributes roughly uniformly", "[rng]", FastRng, RanlxdRng, Mt19937Rng) {
    constexpr int k_bins        = 16;
    constexpr int k_samples     = 320'000;
    constexpr double k_expected = static_cast<double>(k_samples) / k_bins;
    std::array<int, k_bins> counts{};

    TestType r{2025};
    for (int i = 0; i < k_samples; ++i) {
        ++counts[r.uniform_int(k_bins)];
    }
    for (int c : counts) {
        REQUIRE(static_cast<double>(c) == Catch::Approx(k_expected).margin(0.03 * k_expected));
    }
}

TEMPLATE_TEST_CASE(
    "Rng::normal mean and stddev match standard normal", "[rng]", FastRng, RanlxdRng, Mt19937Rng) {
    constexpr int k_samples = 200'000;
    TestType r{99};
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
    REQUIRE(mean == Catch::Approx(0.0).margin(0.02));
    REQUIRE(stddev == Catch::Approx(1.0).margin(0.02));
}

TEMPLATE_TEST_CASE("Rng::normal_fill agrees with sequential normal() calls",
                   "[rng]",
                   FastRng,
                   RanlxdRng,
                   Mt19937Rng) {
    // Same seed → same statistical stream from both paths, but the SIMD
    // Sleef-batched `normal_fill` path differs from sequential scalar
    // `normal()` in the last ULP across ISAs (NEON vs SSE2/AVX2/AVX512).
    // The contract is statistical equivalence at machine precision, not
    // bit identity — see the comment on `Rng::normal_fill`.
    constexpr std::size_t n   = 1024;
    constexpr double k_margin = 1e-12;
    TestType a{123};
    TestType b{123};
    std::vector<double> fill_buf(n);
    a.normal_fill(fill_buf.data(), n);
    for (std::size_t i = 0; i < n; ++i) {
        double const expected = b.normal();
        REQUIRE(fill_buf[i] == Catch::Approx(expected).margin(k_margin));
    }
}

TEMPLATE_TEST_CASE("Copying Rng snapshots state and diverges after reseed",
                   "[rng]",
                   FastRng,
                   RanlxdRng,
                   Mt19937Rng) {
    TestType a{55};
    (void)a.uniform_u64();
    (void)a.uniform_u64();

    TestType b = a;
    for (int i = 0; i < 4; ++i) {
        REQUIRE(a.uniform_u64() == b.uniform_u64());
    }
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
