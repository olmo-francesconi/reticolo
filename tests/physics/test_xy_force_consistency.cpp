#include <reticolo/action/bond/xy.hpp>
#include <reticolo/action/concepts.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng/fast_rng.hpp>
#include <reticolo/core/site.hpp>

#include <cmath>
#include <cstddef>
#include <numbers>

#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::Lattice;
using reticolo::Site;
using reticolo::action::HmcAction;
using reticolo::action::Xy;

static_assert(HmcAction<Xy<double>, Lattice<double>>);

namespace {

void randomize_angles(Lattice<double>& theta, FastRng& rng) {
    constexpr double k_two_pi = 2.0 * std::numbers::pi;
    for (Site const x : theta.sites()) {
        theta[x] = k_two_pi * rng.uniform();
    }
}

}  // namespace

TEST_CASE("Xy: compute_force matches central FD of s_full", "[physics][xy]") {
    Xy<double> const action{.beta = 0.9};

    Lattice<double> theta{{6, 6}};
    Lattice<double> force{theta.indexing()};
    FastRng rng{29};
    randomize_angles(theta, rng);

    action.compute_force(theta, force);

    constexpr double k_eps = 1e-4;
    constexpr double k_tol = 1e-6;

    for (std::size_t trial = 0; trial < 25; ++trial) {
        Site const x     = Site{rng.uniform_int(theta.nsites())};
        double const old = theta[x];

        theta[x]             = old + k_eps;
        double const s_plus  = action.s_full(theta);
        theta[x]             = old - k_eps;
        double const s_minus = action.s_full(theta);
        theta[x]             = old;

        double const grad_numeric = (s_plus - s_minus) / (2.0 * k_eps);
        REQUIRE(std::abs(force[x] - (-grad_numeric)) < k_tol);
    }
}

TEST_CASE("Xy at beta=0 yields zero action and zero force", "[physics][xy]") {
    Xy<double> const action{.beta = 0.0};

    Lattice<double> theta{{4, 4}};
    Lattice<double> force{theta.indexing()};
    FastRng rng{0};
    randomize_angles(theta, rng);

    REQUIRE(action.s_full(theta) == 0.0);
    action.compute_force(theta, force);
    for (Site const x : theta.sites()) {
        REQUIRE(force[x] == 0.0);
    }
}
