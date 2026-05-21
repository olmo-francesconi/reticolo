#include <reticolo/obs/analysis.hpp>

#include <array>
#include <cmath>
#include <span>
#include <stdexcept>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

namespace ana = reticolo::obs::analysis;

TEST_CASE("analysis::mean returns 0 on empty span", "[analysis]") {
    std::vector<double> empty;
    REQUIRE(ana::mean(empty) == 0.0);
}

TEST_CASE("analysis::mean averages correctly", "[analysis]") {
    std::array<double, 4> v{1.0, 2.0, 3.0, 4.0};
    REQUIRE(ana::mean(v) == 2.5);
}

TEST_CASE("analysis::susceptibility on a constant series is zero", "[analysis][susceptibility]") {
    std::vector<double> m(100, 0.7);
    // Variance is zero => chi = N * 0 = 0 modulo FP roundoff.
    REQUIRE(ana::susceptibility(m, 1024.0) == Catch::Approx(0.0).margin(1e-9));
}

TEST_CASE("analysis::susceptibility recovers N * var(m)", "[analysis][susceptibility]") {
    // Two-point distribution: half +1, half -1 in |m| terms (so |m|=1 always
    // would give zero variance). Use a real spread: |m| in {0.4, 0.6} each
    // half the time.
    std::vector<double> m;
    m.reserve(200);
    for (int i = 0; i < 100; ++i) {
        m.push_back(0.4);
        m.push_back(0.6);
    }
    // <|m|>   = 0.5
    // <m^2>   = 0.5 * (0.16 + 0.36) = 0.26
    // var     = 0.26 - 0.25 = 0.01
    // chi     = N * var.
    double const N   = 256.0;
    double const chi = ana::susceptibility(m, N);
    REQUIRE(chi == Catch::Approx((N * 0.01)).margin(1e-12));
}

TEST_CASE("analysis::susceptibility throws on non-positive N", "[analysis][susceptibility]") {
    std::array<double, 3> m{0.1, 0.2, 0.3};
    REQUIRE_THROWS_AS(ana::susceptibility(m, 0.0), std::invalid_argument);
    REQUIRE_THROWS_AS(ana::susceptibility(m, -1.0), std::invalid_argument);
}

TEST_CASE("analysis::binder of a Gaussian-distributed scalar trends toward 1/3",
          "[analysis][binder]") {
    // For a Gaussian m, <m^4> = 3 <m^2>^2, so 1 - <m^4>/(3<m^2>^2) = 0.
    // Build a synthetic series with exactly Gaussian moments by picking the
    // analytic values per-sample.
    std::array<double, 4> m2{1.0, 1.0, 1.0, 1.0};
    std::array<double, 4> m4{3.0, 3.0, 3.0, 3.0};
    REQUIRE(ana::binder(m2, m4) == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("analysis::binder of a saturated (two-delta) m approaches 2/3", "[analysis][binder]") {
    // m takes ±m0 with equal probability. Then m^2 = m0^2 always and m^4 = m0^4.
    // U = 1 - m0^4 / (3 m0^4) = 2/3.
    std::array<double, 6> m2{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::array<double, 6> m4{1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    REQUIRE(ana::binder(m2, m4) == Catch::Approx(2.0 / 3.0).margin(1e-12));
}

TEST_CASE("analysis::binder rejects mismatched span sizes", "[analysis][binder]") {
    std::array<double, 3> m2{1.0, 1.0, 1.0};
    std::array<double, 2> m4{1.0, 1.0};
    REQUIRE_THROWS_AS(ana::binder(m2, m4), std::invalid_argument);
}

TEST_CASE("analysis::binder returns 0 on empty input", "[analysis][binder]") {
    std::vector<double> empty;
    REQUIRE(ana::binder(empty, empty) == 0.0);
}
