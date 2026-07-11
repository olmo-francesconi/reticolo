#include <reticolo/action/nn/phi4.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng/fast_rng.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/updater/hmc/hmc.hpp>
#include <reticolo/updater/hmc/integrators.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::Lattice;
using reticolo::Site;
using reticolo::action::Phi4;
using reticolo::updater::Hmc;
using reticolo::updater::integ::Leapfrog;
using reticolo::updater::integ::Omelyan2;
using reticolo::updater::integ::Omelyan4;

template <class Integrator>
static void check_reversibility() {
    Phi4<double> const action{.kappa = 0.18, .lambda = 0.04};

    Lattice<double> phi{{4, 4, 4}};
    FastRng rng{31415};
    for (Site x : phi.sites()) {
        phi[x] = rng.normal();
    }

    Hmc hmc{action, phi, rng, {.tau = 1.0, .n_md = 20}, Integrator{}};

    for (Site x : phi.sites()) {
        hmc.momentum()[x] = rng.normal();
    }
    std::vector<double> q0(phi.nsites());
    std::vector<double> p0(phi.nsites());
    for (Site x : phi.sites()) {
        q0[x.value()] = phi[x];
        p0[x.value()] = hmc.momentum()[x];
    }

    hmc.integrate_only(1.0, 20);

    for (Site x : phi.sites()) {
        hmc.momentum()[x] = -hmc.momentum()[x];
    }
    hmc.integrate_only(1.0, 20);

    double max_q_err = 0.0;
    double max_p_err = 0.0;
    for (Site x : phi.sites()) {
        max_q_err = std::max(max_q_err, std::abs(phi[x] - q0[x.value()]));
        max_p_err = std::max(max_p_err, std::abs(hmc.momentum()[x] - (-p0[x.value()])));
    }
    REQUIRE(max_q_err < 1e-10);
    REQUIRE(max_p_err < 1e-10);
}

TEST_CASE("HMC Leapfrog is reversible to ~1e-10", "[physics][hmc][reversibility]") {
    check_reversibility<Leapfrog>();
}

TEST_CASE("HMC Omelyan2 is reversible to ~1e-10", "[physics][hmc][reversibility]") {
    check_reversibility<Omelyan2>();
}

TEST_CASE("HMC Omelyan4 is reversible to ~1e-10", "[physics][hmc][reversibility]") {
    check_reversibility<Omelyan4>();
}

TEST_CASE("HMC trajectory is symplectic: dH small and bounded vs trajectory length",
          "[physics][hmc][reversibility]") {
    Phi4<double> const action{.kappa = 0.15, .lambda = 0.03};

    Lattice<double> phi{{4, 4, 4}};
    FastRng rng{271828};
    for (Site x : phi.sites()) {
        phi[x] = 0.1 * rng.normal();
    }

    Hmc hmc{action, phi, rng, {.tau = 0.5, .n_md = 50}, Leapfrog{}};

    // dH should hover near zero with no secular drift; we tolerate |dH| < ~0.5
    // ensemble-wise. The point of this test is "no obvious bug producing big dH".
    double max_abs_dh = 0.0;
    for (int t = 0; t < 32; ++t) {
        auto step  = hmc.step();
        max_abs_dh = std::max(max_abs_dh, std::abs(step.dH));
    }
    REQUIRE(max_abs_dh < 1.0);
}
