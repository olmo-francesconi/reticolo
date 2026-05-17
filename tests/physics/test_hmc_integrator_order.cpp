#include <reticolo/action/builtins/phi4.hpp>
#include <reticolo/algorithm/hmc.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <array>
#include <cmath>
#include <cstddef>

#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::Lattice;
using reticolo::Site;
using reticolo::action::Phi4;
using reticolo::alg::Hmc;
using reticolo::alg::HmcSpec;
using reticolo::alg::integ::Leapfrog;

// Returns ensemble-mean |dH| over `n_traj` trajectories of (tau, n_md).
// Each trajectory uses fresh Gaussian momenta drawn from `rng`.
template <class A>
static double mean_abs_dh(A const& action,
                          Lattice<double>& phi,
                          FastRng& rng,
                          double const tau,
                          int const n_md,
                          int const n_traj) {
    Hmc<A, FastRng, Leapfrog> hmc{action, phi, rng, {.tau = tau, .n_md = n_md}};
    double sum = 0.0;
    for (int t = 0; t < n_traj; ++t) {
        auto step = hmc.trajectory();
        sum += std::abs(step.dH);
    }
    return sum / static_cast<double>(n_traj);
}

TEST_CASE("HMC Leapfrog has integrator order p ~ 2 in |dH| vs dt", "[physics][hmc][order]") {
    Phi4<double> const action{.kappa = 0.13, .lambda = 0.02};

    constexpr double k_tau               = 1.0;
    constexpr std::array<int, 4> k_n_mds = {8, 16, 32, 64};
    constexpr int k_n_traj               = 64;
    std::array<double, k_n_mds.size()> mean_dh{};
    std::array<double, k_n_mds.size()> dt{};

    for (std::size_t i = 0; i < k_n_mds.size(); ++i) {
        // Same starting field + seed for every dt → all comparisons see the same
        // (q0, p0) draws. This isolates the integrator-error scaling.
        Lattice<double> phi{{4, 4, 4}};
        FastRng rng{11223344};
        for (Site x : phi.sites()) {
            phi[x] = 0.1 * rng.normal();
        }
        mean_dh[i] = mean_abs_dh(action, phi, rng, k_tau, k_n_mds[i], k_n_traj);
        dt[i]      = k_tau / static_cast<double>(k_n_mds[i]);
    }

    // Log-log linear regression slope.
    double sum_x  = 0.0;
    double sum_y  = 0.0;
    double sum_xx = 0.0;
    double sum_xy = 0.0;
    for (std::size_t i = 0; i < k_n_mds.size(); ++i) {
        double const x = std::log(dt[i]);
        double const y = std::log(mean_dh[i]);
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    double const n     = static_cast<double>(k_n_mds.size());
    double const slope = ((n * sum_xy) - (sum_x * sum_y)) / ((n * sum_xx) - (sum_x * sum_x));

    INFO("dt    : " << dt[0] << ", " << dt[1] << ", " << dt[2] << ", " << dt[3]);
    INFO("dH    : " << mean_dh[0] << ", " << mean_dh[1] << ", " << mean_dh[2] << ", "
                    << mean_dh[3]);
    INFO("slope = " << slope);
    REQUIRE(slope > 1.7);
    REQUIRE(slope < 2.3);
}
