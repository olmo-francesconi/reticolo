#include <reticolo/action/phi4.hpp>
#include <reticolo/algorithm/metropolis.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <cmath>

#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::Lattice;
using reticolo::Site;
using reticolo::action::Phi4;
using reticolo::alg::Metropolis;

// With kappa = 0 and lambda = 0 the action factorises into independent on-site
// Gaussians: exp(-phi(x)^2) per site, i.e. phi ~ N(0, sigma=1/sqrt(2)). The
// equilibrium expectation values are then:
//   <phi>    = 0
//   <phi^2>  = 1/2
//
// A Metropolis sweep on this action must reproduce both to within a few
// percent in modest statistics.
TEST_CASE("Metropolis on decoupled action reproduces N(0, 1/sqrt(2))",
          "[physics][metropolis][free-limit]") {
    Phi4<double> const action{.kappa = 0.0, .lambda = 0.0};

    Lattice<double> phi{{8, 8, 8}};
    FastRng rng{20260517};
    // Hot start.
    for (Site x : phi.sites()) {
        phi[x] = rng.normal();
    }

    Metropolis<Phi4<double>, FastRng> mc{action, phi, rng, /*sigma=*/1.0};

    // Thermalise (decoupled => fast).
    for (int s = 0; s < 200; ++s) {
        (void)mc.sweep();
    }

    constexpr int n_meas  = 400;
    double sum_phi        = 0.0;
    double sum_phi2       = 0.0;
    std::size_t acc_total = 0;
    std::size_t att_total = 0;
    for (int meas = 0; meas < n_meas; ++meas) {
        auto stats = mc.sweep();
        acc_total += stats.accepted;
        att_total += stats.attempts;
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
