#include <reticolo/action/concepts.hpp>
#include <reticolo/action/nn/phi4.hpp>
#include <reticolo/action/nn/phi6.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng/fast_rng.hpp>
#include <reticolo/core/site.hpp>

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
using reticolo::action::Phi6;

static_assert(HmcAction<Phi6<double>, Lattice<double>>);
static_assert(HasFusedKick<Phi6<double>, Lattice<double>>);

namespace {

void randomize(Lattice<double>& phi, FastRng& rng) {
    for (Site const x : phi.sites()) {
        phi[x] = 0.5 * rng.normal();
    }
}

}  // namespace

TEST_CASE("Phi6: compute_force matches central FD of s_full", "[physics][phi6]") {
    Phi6<double> const action{.kappa = 0.15, .lambda = 0.03, .g6 = 0.01};

    Lattice<double> phi{{6, 6, 6}};
    Lattice<double> force{phi.indexing()};
    FastRng rng{271828};
    randomize(phi, rng);

    action.compute_force(phi, force);

    constexpr double k_eps = 1e-4;
    constexpr double k_tol = 1e-6;

    for (std::size_t trial = 0; trial < 25; ++trial) {
        Site const x     = Site{rng.uniform_int(phi.nsites())};
        double const old = phi[x];

        phi[x]              = old + k_eps;
        double const s_plus = action.s_full(phi);

        phi[x]               = old - k_eps;
        double const s_minus = action.s_full(phi);

        phi[x]                       = old;
        double const grad_numeric    = (s_plus - s_minus) / (2.0 * k_eps);
        double const force_predicted = force[x];

        REQUIRE(force_predicted == Catch::Approx((-grad_numeric)).margin(k_tol));
    }
}

TEST_CASE("Phi6 at g6=0 reduces exactly to Phi4", "[physics][phi6]") {
    Phi6<double> const a6{.kappa = 0.18, .lambda = 0.04, .g6 = 0.0};
    Phi4<double> const a4{.kappa = 0.18, .lambda = 0.04};

    Lattice<double> phi{{4, 4, 4}};
    FastRng rng{42};
    randomize(phi, rng);

    REQUIRE(a6.s_full(phi) == Catch::Approx(a4.s_full(phi)).margin(1e-12));

    Lattice<double> f6{phi.indexing()};
    Lattice<double> f4{phi.indexing()};
    a6.compute_force(phi, f6);
    a4.compute_force(phi, f4);
    for (Site const x : phi.sites()) {
        REQUIRE(f6[x] == Catch::Approx(f4[x]).margin(1e-12));
    }
}
