#include <reticolo/action/nn/phi4.hpp>
#include <reticolo/core/field/lattice.hpp>
#include <reticolo/core/rng/fast_rng.hpp>
#include <reticolo/core/field/site.hpp>
#include <reticolo/updater/hmc/hmc.hpp>
#include <reticolo/updater/hmc/integrators.hpp>

#include <array>
#include <cmath>
#include <cstddef>

#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::Lattice;
using reticolo::Site;
using reticolo::action::Phi4;
using reticolo::updater::Hmc;
using reticolo::updater::integ::Leapfrog;
using reticolo::updater::integ::Omelyan2;
using reticolo::updater::integ::Omelyan4;

// Returns ensemble-mean |dH| over `n_traj` trajectories of (tau, n_md).
// Each trajectory uses fresh Gaussian momenta drawn from `rng`.
template <class Integrator, class A>
static double mean_abs_dh(A const& action,
                          Lattice<double>& phi,
                          FastRng& rng,
                          double const tau,
                          int const n_md,
                          int const n_traj) {
    Hmc hmc{action, phi, rng, {.tau = tau, .n_md = n_md}, Integrator{}};
    double sum = 0.0;
    for (int t = 0; t < n_traj; ++t) {
        auto step = hmc.step();
        sum += std::abs(step.dH);
    }
    return sum / static_cast<double>(n_traj);
}

template <std::size_t N>
static double dh_slope(std::array<double, N> const& dt, std::array<double, N> const& mean_dh) {
    double sum_x  = 0.0;
    double sum_y  = 0.0;
    double sum_xx = 0.0;
    double sum_xy = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        double const x = std::log(dt[i]);
        double const y = std::log(mean_dh[i]);
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    auto const n = static_cast<double>(N);
    return ((n * sum_xy) - (sum_x * sum_y)) / ((n * sum_xx) - (sum_x * sum_x));
}

template <class Integrator, std::size_t N>
static double measure_order(Phi4<double> const& action,
                            double const tau,
                            std::array<int, N> const& n_mds,
                            int const n_traj) {
    std::array<double, N> mean_dh{};
    std::array<double, N> dt{};
    for (std::size_t i = 0; i < N; ++i) {
        // Same starting field + seed for every dt → all comparisons see the
        // same (q0, p0) draws. This isolates the integrator-error scaling.
        Lattice<double> phi{{4, 4, 4}};
        FastRng rng{11223344};
        for (Site x : phi.sites()) {
            phi[x] = 0.1 * rng.normal();
        }
        mean_dh[i] = mean_abs_dh<Integrator>(action, phi, rng, tau, n_mds[i], n_traj);
        dt[i]      = tau / static_cast<double>(n_mds[i]);
    }
    INFO("dt    : " << dt[0] << ", " << dt[1] << ", " << dt[2] << ", " << dt[3]);
    INFO("dH    : " << mean_dh[0] << ", " << mean_dh[1] << ", " << mean_dh[2] << ", "
                    << mean_dh[3]);
    return dh_slope(dt, mean_dh);
}

TEST_CASE("HMC Leapfrog has integrator order p ~ 2 in |dH| vs dt", "[physics][hmc][order]") {
    Phi4<double> const action{.kappa = 0.13, .lambda = 0.02};
    constexpr std::array<int, 4> const k_n_mds = {8, 16, 32, 64};
    double const slope = measure_order<Leapfrog>(action, /*tau=*/1.0, k_n_mds, /*n_traj=*/64);
    INFO("Leapfrog slope = " << slope);
    REQUIRE(slope > 1.7);
    REQUIRE(slope < 2.3);
}

TEST_CASE("HMC Omelyan2 has integrator order p ~ 2 in |dH| vs dt", "[physics][hmc][order]") {
    Phi4<double> const action{.kappa = 0.13, .lambda = 0.02};
    constexpr std::array<int, 4> const k_n_mds = {8, 16, 32, 64};
    double const slope = measure_order<Omelyan2>(action, /*tau=*/1.0, k_n_mds, /*n_traj=*/64);
    INFO("Omelyan2 slope = " << slope);
    REQUIRE(slope > 1.7);
    REQUIRE(slope < 2.3);
}

TEST_CASE("HMC Omelyan4 has integrator order p ~ 4 in |dH| vs dt", "[physics][hmc][order]") {
    Phi4<double> const action{.kappa = 0.13, .lambda = 0.02};
    // Order-4 |dH| collapses fast — keep dt larger so we stay clear of the
    // floating-point noise floor at the small end.
    constexpr std::array<int, 4> const k_n_mds = {4, 6, 8, 12};
    double const slope = measure_order<Omelyan4>(action, /*tau=*/1.0, k_n_mds, /*n_traj=*/64);
    INFO("Omelyan4 slope = " << slope);
    REQUIRE(slope > 3.5);
    REQUIRE(slope < 4.5);
}
