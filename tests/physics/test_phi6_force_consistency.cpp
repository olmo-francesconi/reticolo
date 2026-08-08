#include <reticolo/action/concepts.hpp>
#include <reticolo/action/nn/phi4.hpp>
#include <reticolo/action/nn/phi6.hpp>

#include "force_fd_helpers.hpp"

#include <cstddef>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::Lattice;
using reticolo::Site;
using reticolo::action::HasFusedKick;
using reticolo::action::HmcAction;
using reticolo::action::Phi4;
using reticolo::action::Phi6;
using reticolo::test::randomize;
using reticolo::test::require_force_consistent;

static_assert(HmcAction<Phi6<double>, Lattice<double>>);
static_assert(HasFusedKick<Phi6<double>, Lattice<double>>);

TEST_CASE("Phi6: compute_force matches central FD of s_full", "[physics][phi6]") {
    Phi6<double> const action{.kappa = 0.15, .lambda = 0.03, .g6 = 0.01};

    Lattice<double> phi{{6, 6, 6}};
    FastRng rng{271828};
    randomize(phi, rng);

    require_force_consistent(action, phi, rng, {.tol = 1e-6});
}

TEST_CASE("Phi6 at g6=0 reduces exactly to Phi4", "[physics][phi6]") {
    Phi6<double> const a6{.kappa = 0.18, .lambda = 0.04, .g6 = 0.0};
    Phi4<double> const a4{.kappa = 0.18, .lambda = 0.04};

    Lattice<double> phi{{4, 4, 4}};
    FastRng rng{42};
    randomize(phi, rng);

    REQUIRE(a6.s_full(phi) == Catch::Approx(a4.s_full(phi)).margin(1e-12));

    Lattice<double> f6{phi.shape()};
    Lattice<double> f4{phi.shape()};
    a6.compute_force(phi, f6);
    a4.compute_force(phi, f4);
    for (Site const x : phi.sites()) {
        REQUIRE(f6[x] == Catch::Approx(f4[x]).margin(1e-12));
    }
}
