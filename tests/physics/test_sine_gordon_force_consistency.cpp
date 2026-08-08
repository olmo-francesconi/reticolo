#include <reticolo/action/concepts.hpp>
#include <reticolo/action/nn/phi4.hpp>
#include <reticolo/action/nn/sine_gordon.hpp>

#include "force_fd_helpers.hpp"

#include <cmath>
#include <cstddef>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::Lattice;
using reticolo::Site;
using reticolo::action::HasFusedKick;
using reticolo::action::HmcAction;
using reticolo::action::Phi4;
using reticolo::action::SineGordon;
using reticolo::test::randomize;
using reticolo::test::require_force_consistent;

static_assert(HmcAction<SineGordon<double>, Lattice<double>>);
static_assert(HasFusedKick<SineGordon<double>, Lattice<double>>);

TEST_CASE("SineGordon: compute_force matches central FD of s_full", "[physics][sine_gordon]") {
    SineGordon<double> const action{.kappa = 0.17, .alpha = 0.4};

    Lattice<double> phi{{6, 6, 6}};
    FastRng rng{2718};
    randomize(phi, rng);

    require_force_consistent(action, phi, rng, {.tol = 1e-6});
}

TEST_CASE("SineGordon at alpha=0 reduces to Phi4 with lambda=0", "[physics][sine_gordon]") {
    SineGordon<double> const sg{.kappa = 0.18, .alpha = 0.0};
    Phi4<double> const p4{.kappa = 0.18, .lambda = 0.0};

    Lattice<double> phi{{4, 4, 4}};
    FastRng rng{42};
    randomize(phi, rng);

    REQUIRE(sg.s_full(phi) == Catch::Approx(p4.s_full(phi)).margin(1e-12));

    Lattice<double> f_sg{phi.shape()};
    Lattice<double> f_p4{phi.shape()};
    sg.compute_force(phi, f_sg);
    p4.compute_force(phi, f_p4);
    for (Site const x : phi.sites()) {
        REQUIRE(f_sg[x] == Catch::Approx(f_p4[x]).margin(1e-12));
    }
}
