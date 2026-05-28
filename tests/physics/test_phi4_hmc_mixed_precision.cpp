#include <reticolo/action/phi4.hpp>
#include <reticolo/algorithm/hmc.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/obs/catalog.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::Lattice;
using reticolo::Site;
using reticolo::action::Phi4;
using reticolo::alg::Hmc;
using reticolo::alg::HmcSpec;

// The S-reduction always returns double, whatever the lattice precision — the
// load-bearing invariant of mixed-precision HMC.
static_assert(std::is_same_v<decltype(std::declval<Phi4<float> const&>().s_full(
                                 std::declval<Lattice<float> const&>())),
                             double>);
static_assert(std::is_same_v<decltype(std::declval<Phi4<double> const&>().s_full(
                                 std::declval<Lattice<double> const&>())),
                             double>);

template <class T>
static void randomize(Lattice<T>& phi, FastRng& rng) {
    for (Site x : phi.sites()) {
        phi[x] = static_cast<T>(rng.normal());
    }
}

TEST_CASE("Phi4<float>: compute_force matches central finite difference of s_full",
          "[physics][phi4][mixed-precision]") {
    Phi4<float> const action{.kappa = 0.18F, .lambda = 0.04F};

    Lattice<float> phi{{6, 6, 6}};
    Lattice<float> force{phi.indexing()};
    FastRng rng{56789};
    randomize(phi, rng);

    action.compute_force(phi, force);

    constexpr float k_eps = 1.0e-2F;  // larger eps than the double test: float storage
    for (std::size_t trial = 0; trial < 25; ++trial) {
        Site const x    = Site{rng.uniform_int(phi.nsites())};
        float const old = phi[x];

        phi[x]              = old + k_eps;
        double const s_plus = action.s_full(phi);

        phi[x]               = old - k_eps;
        double const s_minus = action.s_full(phi);

        phi[x] = old;

        double const grad_numeric    = (s_plus - s_minus) / (2.0 * static_cast<double>(k_eps));
        double const force_predicted = static_cast<double>(force[x]);
        double const force_numeric   = -grad_numeric;

        // Single-precision per-site terms + O(eps^2) central-diff truncation:
        // a percent-level agreement, not the 1e-7 of the double kernel.
        REQUIRE(force_predicted == Catch::Approx(force_numeric).epsilon(0.02).margin(0.02));
    }
}

// Float and double HMC must sample the same equilibrium <phi^2>. The float
// trajectory integrates in single precision, but the acceptance test uses the
// double-accumulated ΔH, so the chain still targets exp(-S).
TEST_CASE("Phi4 HMC: float and double agree on equilibrium <phi^2>",
          "[physics][phi4][hmc][mixed-precision]") {
    constexpr double k_kappa  = 0.12;
    constexpr double k_lambda = 1.0;

    auto equilibrium_phi2 = [](auto scalar_tag, unsigned long long seed, double& acc_out) {
        using T = decltype(scalar_tag);
        Lattice<T> phi{{8, 8, 8}};
        FastRng rng{seed};
        Phi4<T> const action{.kappa = static_cast<T>(k_kappa), .lambda = static_cast<T>(k_lambda)};
        Hmc hmc{action, phi, rng, HmcSpec{.tau = 1.0, .n_md = 20}};

        for (int i = 0; i < 300; ++i) {
            (void)hmc.step(reticolo::log::Mode::silent);
        }
        constexpr int n_meas = 800;
        double sum2          = 0.0;
        long accepted        = 0;
        for (int m = 0; m < n_meas; ++m) {
            accepted += hmc.step(reticolo::log::Mode::silent).accepted ? 1 : 0;
            sum2 += reticolo::obs::sq(phi);
        }
        acc_out = static_cast<double>(accepted) / static_cast<double>(n_meas);
        return sum2 / static_cast<double>(n_meas);
    };

    double acc_d        = 0.0;
    double acc_f        = 0.0;
    double const phi2_d = equilibrium_phi2(double{}, 2024, acc_d);
    double const phi2_f = equilibrium_phi2(float{}, 4048, acc_f);

    INFO("<phi^2> double = " << phi2_d << "  (acc " << acc_d << ")");
    INFO("<phi^2> float  = " << phi2_f << "  (acc " << acc_f << ")");

    // Single-precision MD still reaches a high acceptance with the double ΔH.
    REQUIRE(acc_f > 0.5);
    REQUIRE(std::abs(phi2_f - phi2_d) < 0.04 * phi2_d);
}
