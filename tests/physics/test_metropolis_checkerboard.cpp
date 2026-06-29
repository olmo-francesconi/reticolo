#include <reticolo/action/phi4.hpp>
#include <reticolo/action/xy.hpp>
#include <reticolo/algorithm/metropolis.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <cmath>
#include <cstddef>

#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::Lattice;
using reticolo::Site;
using reticolo::action::Phi4;
using reticolo::action::Xy;
using reticolo::alg::Metropolis;
using reticolo::alg::MetropolisSpec;
using reticolo::alg::Sweep;

// Checkerboard is a distinct (still valid) Markov chain from the sequential
// sweep, so the contract is that it samples the SAME equilibrium distribution.
//
// With kappa = lambda = 0 the action factorises into independent on-site
// Gaussians exp(-phi^2): phi ~ N(0, 1/sqrt(2)), so <phi> = 0 and <phi^2> = 1/2.
TEST_CASE("Checkerboard Metropolis reproduces N(0, 1/sqrt(2))",
          "[physics][metropolis][checkerboard][free-limit]") {
    Phi4<double> const action{.kappa = 0.0, .lambda = 0.0};

    Lattice<double> phi{{8, 8, 8}};
    FastRng rng{20260528};
    for (Site x : phi.sites()) {
        phi[x] = rng.normal();
    }

    Metropolis<Phi4<double>, FastRng> mc{
        action, phi, rng, MetropolisSpec{.sigma = 1.0, .sweep = Sweep::Checkerboard}};

    for (int s = 0; s < 200; ++s) {
        (void)mc.step();
    }

    constexpr int n_meas  = 400;
    double sum_phi        = 0.0;
    double sum_phi2       = 0.0;
    std::size_t acc_total = 0;
    std::size_t att_total = 0;
    for (int meas = 0; meas < n_meas; ++meas) {
        auto stats = mc.step();
        acc_total += stats.n_accepted;
        att_total += stats.n_attempts;
        for (Site x : phi.sites()) {
            sum_phi += phi[x];
            sum_phi2 += phi[x] * phi[x];
        }
    }

    double const n_samples = static_cast<double>(n_meas) * static_cast<double>(phi.nsites());
    double const mean      = sum_phi / n_samples;
    double const mean_sq   = sum_phi2 / n_samples;
    double const acc_rate  = static_cast<double>(acc_total) / static_cast<double>(att_total);

    INFO("<phi>   = " << mean << "  (expect 0)");
    INFO("<phi^2> = " << mean_sq << "  (expect 0.5)");
    INFO("acc     = " << acc_rate);

    REQUIRE(std::abs(mean) < 0.02);
    REQUIRE(std::abs(mean_sq - 0.5) < 0.02);
    REQUIRE(acc_rate > 0.2);
    REQUIRE(acc_rate < 0.95);
}

// At finite coupling the two sweeps must agree on equilibrium <phi^2>. Run each
// to equilibrium from an independent seed and compare the estimators.
TEST_CASE("Checkerboard and sequential agree on equilibrium <phi^2>",
          "[physics][metropolis][checkerboard]") {
    Phi4<double> const action{.kappa = 0.10, .lambda = 0.5};

    auto equilibrium_phi2 = [&](Sweep sweep, unsigned long long seed) {
        Lattice<double> phi{{8, 8, 8}};
        FastRng rng{seed};
        for (Site x : phi.sites()) {
            phi[x] = 0.1 * rng.normal();
        }
        Metropolis<Phi4<double>, FastRng> mc{
            action, phi, rng, MetropolisSpec{.sigma = 0.7, .sweep = sweep}};
        for (int s = 0; s < 500; ++s) {
            (void)mc.step();
        }
        constexpr int n_meas = 600;
        double sum2          = 0.0;
        for (int meas = 0; meas < n_meas; ++meas) {
            (void)mc.step();
            for (Site x : phi.sites()) {
                sum2 += phi[x] * phi[x];
            }
        }
        return sum2 / (static_cast<double>(n_meas) * static_cast<double>(phi.nsites()));
    };

    double const seq = equilibrium_phi2(Sweep::Sequential, 111);
    double const chk = equilibrium_phi2(Sweep::Checkerboard, 222);

    INFO("<phi^2> sequential   = " << seq);
    INFO("<phi^2> checkerboard = " << chk);
    REQUIRE(std::abs(seq - chk) < 0.04 * seq);
}

// Requesting checkerboard on an action with no NN ds_local_from_nbrs (Xy) must
// not break: the updater logs once and runs the sequential generic path.
TEST_CASE("Checkerboard on an action without an NN fast path falls back",
          "[metropolis][checkerboard][fallback]") {
    Xy<double> const action{.beta = 0.5};

    Lattice<double> phi{{8, 8}};
    FastRng rng{7};
    for (Site x : phi.sites()) {
        phi[x] = rng.uniform() * 6.283185307179586;
    }

    Metropolis<Xy<double>, FastRng> mc{
        action, phi, rng, MetropolisSpec{.sigma = 0.5, .sweep = Sweep::Checkerboard}};

    std::size_t acc = 0;
    std::size_t att = 0;
    for (int s = 0; s < 50; ++s) {
        auto stats = mc.step();
        acc += stats.n_accepted;
        att += stats.n_attempts;
    }

    REQUIRE(att == 50 * phi.nsites());
    REQUIRE(acc <= att);
    REQUIRE(acc > 0);
}
