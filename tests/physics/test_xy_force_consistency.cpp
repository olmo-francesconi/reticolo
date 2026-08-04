#include <reticolo/action/concepts.hpp>
#include <reticolo/action/nn/xy.hpp>

#include "force_fd_helpers.hpp"

#include <cmath>
#include <cstddef>
#include <numbers>

#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::Lattice;
using reticolo::Site;
using reticolo::action::HmcAction;
using reticolo::action::Xy;
using reticolo::test::require_force_consistent;

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
    FastRng rng{29};
    randomize_angles(theta, rng);

    require_force_consistent(action, theta, rng, {.tol = 1e-6});
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
