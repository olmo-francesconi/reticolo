#include <reticolo/action/detail/concepts.hpp>
#include <reticolo/action/site/phi4.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <cmath>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::Lattice;
using reticolo::Site;
using reticolo::action::HasFusedKick;
using reticolo::action::HmcAction;
using reticolo::action::Phi4;

static_assert(HmcAction<Phi4<double>, Lattice<double>>);
static_assert(HasFusedKick<Phi4<double>, Lattice<double>>);

// Seed a small lattice with iid N(0, 1) values.
template <class T>
static void randomize(Lattice<T>& phi, FastRng& rng) {
    for (Site x : phi.sites()) {
        phi[x] = static_cast<T>(rng.normal());
    }
}

TEST_CASE("Phi4: compute_force matches central finite difference of s_full", "[physics][phi4]") {
    Phi4<double> const action{.kappa = 0.18, .lambda = 0.04};

    Lattice<double> phi{{6, 6, 6}};
    Lattice<double> force{phi.indexing()};
    FastRng rng{56789};
    randomize(phi, rng);

    action.compute_force(phi, force);

    constexpr double k_eps = 1e-4;
    constexpr double k_tol = 1e-7;  // central diff O(eps^2) ~ 1e-8; allow margin

    for (std::size_t trial = 0; trial < 25; ++trial) {
        Site const x     = Site{rng.uniform_int(phi.nsites())};
        double const old = phi[x];

        phi[x]              = old + k_eps;
        double const s_plus = action.s_full(phi);

        phi[x]               = old - k_eps;
        double const s_minus = action.s_full(phi);

        phi[x] = old;

        double const grad_numeric    = (s_plus - s_minus) / (2.0 * k_eps);
        double const force_predicted = force[x];
        double const force_numeric   = -grad_numeric;

        REQUIRE(force_predicted == Catch::Approx(force_numeric).margin(k_tol));
    }
}

TEST_CASE("Phi4: free-theory limit (lambda=0) gives force = 2 kappa sum_nn - 2 phi",
          "[physics][phi4]") {
    Phi4<double> const action{.kappa = 0.15, .lambda = 0.0};

    Lattice<double> phi{{4, 4, 4}};
    Lattice<double> force{phi.indexing()};
    FastRng rng{42};
    randomize(phi, rng);
    action.compute_force(phi, force);

    for (Site x : phi.sites()) {
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
    for (Site x : phi.sites()) {
        s += phi[x] * phi[x];
    }
    REQUIRE(action.s_full(phi) == Catch::Approx(s).margin(1e-12));
}
