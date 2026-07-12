#include <reticolo/action/nn/phi4.hpp>
#include <reticolo/core/field/lattice.hpp>
#include <reticolo/core/rng/fast_rng.hpp>
#include <reticolo/core/field/site.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::Lattice;
using reticolo::Site;
using reticolo::action::Phi4;

// The S-reduction always returns double, whatever the lattice precision — the
// load-bearing invariant of mixed-precision HMC.
static_assert(std::is_same_v<decltype(std::declval<Phi4<float> const&>().s_full(
                                 std::declval<Lattice<float> const&>())),
                             double>);
static_assert(std::is_same_v<decltype(std::declval<Phi4<double> const&>().s_full(
                                 std::declval<Lattice<double> const&>())),
                             double>);

TEST_CASE("Phi4<float>: compute_force matches central finite difference of s_full",
          "[physics][phi4][mixed-precision]") {
    Phi4<float> const action{.kappa = 0.18F, .lambda = 0.04F};

    Lattice<float> phi{{6, 6, 6}};
    Lattice<float> force{phi.indexing()};
    FastRng rng{56789};
    for (Site x : phi.sites()) {
        phi[x] = static_cast<float>(rng.normal());
    }

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

        REQUIRE(force_predicted == Catch::Approx(force_numeric).epsilon(0.02).margin(0.02));
    }
}

// Cheap cross-check: on a single shared configuration, the float kernels must
// reproduce the double kernels to within single precision. No Monte Carlo —
// this validates the mixed-precision plumbing (per-site math in T, double
// reduction) directly on the kernels.
TEST_CASE("Phi4: float kernels reproduce double on one configuration",
          "[physics][phi4][mixed-precision]") {
    Phi4<double> const ad{.kappa = 0.12, .lambda = 1.0};
    Phi4<float> const af{.kappa = 0.12F, .lambda = 1.0F};

    Lattice<double> pd{{8, 8, 8}};
    Lattice<float> pf{pd.shape()};
    FastRng rng{20260601};
    for (Site x : pd.sites()) {
        double const v = rng.normal();
        pd[x]          = v;
        pf[x]          = static_cast<float>(v);
    }

    // s_full: both return double; agree to single precision.
    REQUIRE(af.s_full(pf) == Catch::Approx(ad.s_full(pd)).epsilon(1.0e-4));

    // compute_force: elementwise agreement.
    Lattice<double> fd{pd.indexing()};
    Lattice<float> ff{pf.indexing()};
    ad.compute_force(pd, fd);
    af.compute_force(pf, ff);
    for (Site x : pd.sites()) {
        REQUIRE(static_cast<double>(ff[x]) == Catch::Approx(fd[x]).epsilon(2.0e-3).margin(1.0e-4));
    }
}
