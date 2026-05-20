#include <reticolo/action/detail/concepts.hpp>
#include <reticolo/action/xy.hpp>
#include <reticolo/algorithm/metropolis.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <cmath>
#include <cstddef>
#include <numbers>

#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::Lattice;
using reticolo::Site;
using reticolo::action::HasForce;
using reticolo::action::HasSEff;
using reticolo::action::LocalAction;
using reticolo::action::Xy;
using reticolo::alg::Metropolis;

static_assert(LocalAction<Xy<double>, double>);
static_assert(HasSEff<Xy<double>, double>);
static_assert(HasForce<Xy<double>, double>);

namespace {

void randomize_angles(Lattice<double>& theta, FastRng& rng) {
    constexpr double k_two_pi = 2.0 * std::numbers::pi;
    for (Site const x : theta.sites()) {
        theta[x] = k_two_pi * rng.uniform();
    }
}

}  // namespace

TEST_CASE("Xy: ds_local matches finite difference of s_full", "[physics][xy]") {
    Xy<double> const action{.beta = 0.7};

    Lattice<double> theta{{6, 6}};
    FastRng rng{17};
    randomize_angles(theta, rng);

    for (std::size_t trial = 0; trial < 20; ++trial) {
        Site const x     = Site{rng.uniform_int(theta.nsites())};
        double const old = theta[x];
        double const nv  = old + (0.5 * rng.normal());

        double const ds_predicted = action.ds_local(theta, x, nv);
        double const s_old        = action.s_full(theta);
        theta[x]                  = nv;
        double const s_new        = action.s_full(theta);
        theta[x]                  = old;

        REQUIRE(std::abs(ds_predicted - (s_new - s_old)) < 1e-9);
    }
}

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

// At low temperature (large beta) all spins align: the NN correlation
// <cos(theta_x - theta_{x+mu})> should sit close to 1 after thermalisation.
// This is a basic sanity check that Metropolis + the XY action approach the
// expected ferromagnetic limit.
TEST_CASE("Xy at large beta: Metropolis equilibrates near aligned phase",
          "[physics][xy][metropolis]") {
    Xy<double> const action{.beta = 4.0};

    Lattice<double> theta{{6, 6}};
    FastRng rng{2024};
    // Mildly random start — close enough to aligned that the chain doesn't
    // get stuck flipping a vortex.
    for (Site const x : theta.sites()) {
        theta[x] = 0.1 * rng.normal();
    }

    Metropolis<Xy<double>, FastRng> mc{
        action, theta, rng, reticolo::alg::MetropolisSpec{.sigma = 0.3}};
    for (int s = 0; s < 800; ++s) {
        (void)mc.sweep();
    }

    constexpr int n_meas = 400;
    double sum_cos       = 0.0;
    std::size_t bonds    = 0;
    for (int meas = 0; meas < n_meas; ++meas) {
        (void)mc.sweep();
        for (Site const x : theta.sites()) {
            for (std::size_t mu = 0; mu < theta.ndims(); ++mu) {
                sum_cos += std::cos(theta[x] - theta[theta.next(x, mu)]);
                ++bonds;
            }
        }
    }
    double const mean_cos = sum_cos / static_cast<double>(bonds);
    INFO("<cos(theta_x - theta_y)> = " << mean_cos << " (expected near 1 at beta=4)");
    REQUIRE(mean_cos > 0.85);
}
