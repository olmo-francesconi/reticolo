#include <reticolo/action/concepts.hpp>
#include <reticolo/action/nn/phi4.hpp>

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
using reticolo::test::randomize;
using reticolo::test::require_force_consistent;

static_assert(HmcAction<Phi4<double>, Lattice<double>>);
static_assert(HasFusedKick<Phi4<double>, Lattice<double>>);

TEST_CASE("Phi4: compute_force matches central finite difference of s_full", "[physics][phi4]") {
    Phi4<double> const action{.kappa = 0.18, .lambda = 0.04};
    Lattice<double> phi{{6, 6, 6}};
    FastRng rng{56789};
    randomize(phi, rng);

    require_force_consistent(action, phi, rng);
}

// Closed forms the FD sweep cannot catch: a wrong overall sign or a dropped term
// still differentiates consistently, so pin the free-theory force and the
// zero-coupling action against hand-derived expressions.
TEST_CASE("Phi4: free-theory limit (lambda=0) gives force = 2 kappa sum_nn - 2 phi",
          "[physics][phi4]") {
    Phi4<double> const action{.kappa = 0.15, .lambda = 0.0};

    Lattice<double> phi{{4, 4, 4}};
    Lattice<double> force{phi.indexing()};
    FastRng rng{42};
    randomize(phi, rng);
    action.compute_force(phi, force);

    for (Site const x : phi.sites()) {
        double nbrs = 0.0;
        for (std::size_t mu = 0; mu < phi.ndims(); ++mu) {
            nbrs += phi[phi.next(x, mu)] + phi[phi.prev(x, mu)];
        }
        double const expected = (2.0 * action.kappa * nbrs) - (2.0 * phi[x]);
        REQUIRE(force[x] == Catch::Approx(expected).margin(1e-12));
    }
}

TEST_CASE("Phi4: zero coupling (kappa=0, lambda=0) reduces to phi^2 + 1", "[physics][phi4]") {
    Phi4<double> const action{.kappa = 0.0, .lambda = 0.0};

    Lattice<double> phi{{4, 4}};
    FastRng rng{7};
    randomize(phi, rng);

    double s = 0.0;
    for (Site const x : phi.sites()) {
        s += phi[x] * phi[x];
    }
    REQUIRE(action.s_full(phi) == Catch::Approx(s).margin(1e-12));
}
