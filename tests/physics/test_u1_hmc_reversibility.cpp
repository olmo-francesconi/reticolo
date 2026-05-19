#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/link_lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/gauge/algorithm/hmc.hpp>
#include <reticolo/action/compact_u1.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::LinkLattice;
using reticolo::Site;
using reticolo::gauge::action::CompactU1;
using reticolo::gauge::alg::Hmc;
using reticolo::gauge::alg::integ::Leapfrog;
using reticolo::gauge::alg::integ::Omelyan2;
using reticolo::gauge::alg::integ::Omelyan4;

template <class Integrator>
static void check_reversibility() {
    CompactU1<double> const action{.beta = 1.0};

    LinkLattice<double> links{{4, 4, 4, 4}};
    FastRng rng{31415};
    {
        double* const d     = links.data();
        std::size_t const n = links.nlinks();
        for (std::size_t i = 0; i < n; ++i) {
            d[i] = rng.normal();
        }
    }

    Hmc<CompactU1<double>, FastRng, Integrator> hmc{action, links, rng, {.tau = 1.0, .n_md = 20}};

    {
        double* const mp    = hmc.momentum().data();
        std::size_t const n = hmc.momentum().nlinks();
        for (std::size_t i = 0; i < n; ++i) {
            mp[i] = rng.normal();
        }
    }
    std::size_t const nl = links.nlinks();
    std::vector<double> q0(nl);
    std::vector<double> p0(nl);
    double const* const qp = links.data();
    double const* const pp = hmc.momentum().data();
    for (std::size_t i = 0; i < nl; ++i) {
        q0[i] = qp[i];
        p0[i] = pp[i];
    }

    hmc.integrate_only(1.0, 20);
    {
        double* const mp = hmc.momentum().data();
        for (std::size_t i = 0; i < nl; ++i) {
            mp[i] = -mp[i];
        }
    }
    hmc.integrate_only(1.0, 20);

    double max_q_err        = 0.0;
    double max_p_err        = 0.0;
    double const* const qpf = links.data();
    double const* const ppf = hmc.momentum().data();
    for (std::size_t i = 0; i < nl; ++i) {
        max_q_err = std::max(max_q_err, std::abs(qpf[i] - q0[i]));
        max_p_err = std::max(max_p_err, std::abs(ppf[i] - (-p0[i])));
    }
    REQUIRE(max_q_err < 1e-10);
    REQUIRE(max_p_err < 1e-10);
}

TEST_CASE("Gauge HMC Leapfrog is reversible to ~1e-10", "[physics][gauge][hmc][reversibility]") {
    check_reversibility<Leapfrog>();
}

TEST_CASE("Gauge HMC Omelyan2 is reversible to ~1e-10", "[physics][gauge][hmc][reversibility]") {
    check_reversibility<Omelyan2>();
}

TEST_CASE("Gauge HMC Omelyan4 is reversible to ~1e-10", "[physics][gauge][hmc][reversibility]") {
    check_reversibility<Omelyan4>();
}
