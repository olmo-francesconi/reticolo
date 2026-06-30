#include <reticolo/core/philox.hpp>
#include <reticolo/core/philox_rng.hpp>
#include <reticolo/core/rng.hpp>

#include <cmath>
#include <cstdint>

#include <catch2/catch_test_macros.hpp>

using namespace reticolo;

// The single source of truth for the CPU + GPU RNG: validate the bijection
// against the published Random123 known-answer vectors so a wrong round /
// schedule is caught here (the device path then only needs to prove it matches
// this host result bit-for-bit).
TEST_CASE("philox4x32-10 matches Random123 known-answer vectors", "[rng][philox]") {
    // Random123 kat_vectors.txt: philox4x32 10, ctr=0, key=0.
    Philox4x32::U32x4 const ctr{0U, 0U, 0U, 0U};
    Philox4x32::U32x2 const key{0U, 0U};
    Philox4x32::U32x4 const o = Philox4x32::bijection(ctr, key);
    REQUIRE(o[0] == 0x6627e8d5U);
    REQUIRE(o[1] == 0xe169c58dU);
    REQUIRE(o[2] == 0xbc57ac4cU);
    REQUIRE(o[3] == 0x9b00dbd8U);
}

TEST_CASE("philox uniforms lie in the half-open unit interval and are reproducible",
          "[rng][philox]") {
    double u0 = 0.0;
    double u1 = 0.0;
    philox_uniform2(123, 5, 9, u0, u1);

    double v0 = 0.0;
    double v1 = 0.0;
    philox_uniform2(123, 5, 9, v0, v1);
    REQUIRE(u0 == v0);  // same (seed, traj, index) -> identical bits
    REQUIRE(u1 == v1);

    REQUIRE(u0 >= 0.0);
    REQUIRE(u0 < 1.0);
    REQUIRE(u1 >= 0.0);
    REQUIRE(u1 < 1.0);

    double w0 = 0.0;
    double w1 = 0.0;
    philox_uniform2(123, 5, 10, w0, w1);  // different index
    REQUIRE(u0 != w0);
}

TEST_CASE("philox normals have N(0,1) moments", "[rng][philox]") {
    constexpr long n = 200000;
    double sum       = 0.0;
    double sumsq     = 0.0;
    for (long i = 0; i < n / 2; ++i) {
        double a = 0.0;
        double b = 0.0;
        philox_normal2(2026, 0, static_cast<std::uint64_t>(i), a, b);
        sum += a + b;
        sumsq += (a * a) + (b * b);
    }
    double const mean = sum / n;
    double const var  = (sumsq / n) - (mean * mean);
    REQUIRE(std::abs(mean) < 0.02);
    REQUIRE(std::abs(var - 1.0) < 0.02);
}

TEST_CASE("PhiloxRng satisfies the Rng concept and is well-behaved", "[rng][philox]") {
    static_assert(Rng<PhiloxRng>);

    PhiloxRng rng{42};
    for (int i = 0; i < 100; ++i) {
        double const u = rng.uniform();
        REQUIRE(u >= 0.0);
        REQUIRE(u < 1.0);
        REQUIRE(rng.uniform_int(7) < 7U);
    }

    PhiloxRng moments{99};
    constexpr int m = 100000;
    double sum      = 0.0;
    double sumsq    = 0.0;
    for (int i = 0; i < m; ++i) {
        double const x = moments.normal();
        sum += x;
        sumsq += x * x;
    }
    double const mean = sum / m;
    double const var  = (sumsq / m) - (mean * mean);
    REQUIRE(std::abs(mean) < 0.02);
    REQUIRE(std::abs(var - 1.0) < 0.03);
}
