// SU(3) Wilson HMC reversibility for all three integrators. Mirrors the
// SU(2) test: integrate forward, negate momentum, integrate back; the field
// must return to its starting configuration and the momentum to its negated
// value to ~1e-9. Exercises Cayley-Hamilton drift + link-centric force.

#include <reticolo/action/detail/gauge_group/su3.hpp>
#include <reticolo/action/wilson.hpp>
#include <reticolo/algorithm/hmc.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/math/su3_ops.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::MatrixLinkLattice;
using reticolo::action::Wilson;
using reticolo::alg::Hmc;
using reticolo::alg::integ::Leapfrog;
using reticolo::alg::integ::Omelyan2;
using reticolo::alg::integ::Omelyan4;
using reticolo::gauge_group::SU3;

namespace {

void hot_start(MatrixLinkLattice<SU3, double>& u, FastRng& rng) {
    std::size_t const d  = u.ndims();
    std::size_t const ns = u.nsites();
    // Initialise to identity.
    std::size_t const total = d * SU3::n_real_components * ns;
    double* const data      = u.data();
    for (std::size_t i = 0; i < total; ++i) {
        data[i] = 0.0;
    }
    for (std::size_t mu = 0; mu < d; ++mu) {
        double* const blk = u.mu_block_data(mu);
        for (std::size_t s = 0; s < ns; ++s) {
            blk[(0 * ns) + s]  = 1.0;
            blk[(8 * ns) + s]  = 1.0;
            blk[(16 * ns) + s] = 1.0;
        }
    }
    // Drift by a small random algebra element to get non-trivial SU(3).
    std::vector<double> scratch_p(SU3::n_real_components * ns);
    for (std::size_t mu = 0; mu < d; ++mu) {
        reticolo::math::su3::sample_algebra_slab(scratch_p.data(), rng, ns);
        reticolo::math::su3::expi_lmul_slab(u.mu_block_data(mu), scratch_p.data(), 0.5, ns);
    }
}

template <class Integrator>
void check_reversibility() {
    Wilson<SU3, double> const action{.beta = 6.0};
    MatrixLinkLattice<SU3, double> u{{4, 4, 4, 4}};
    FastRng rng{31415};
    hot_start(u, rng);

    Hmc<Wilson<SU3, double>, FastRng, Integrator, MatrixLinkLattice<SU3, double>> hmc{
        action, u, rng, {.tau = 0.3, .n_md = 10}};

    std::size_t const d  = u.ndims();
    std::size_t const ns = u.nsites();
    for (std::size_t mu = 0; mu < d; ++mu) {
        reticolo::math::su3::sample_algebra_slab(hmc.momentum().mu_block_data(mu), rng, ns);
    }

    std::size_t const nflat = d * SU3::n_real_components * ns;
    std::vector<double> u0(nflat);
    std::vector<double> p0(nflat);
    double const* const up = u.data();
    double const* const pp = hmc.momentum().data();
    for (std::size_t i = 0; i < nflat; ++i) {
        u0[i] = up[i];
        p0[i] = pp[i];
    }

    hmc.integrate_only(0.3, 10);

    double* const m = hmc.momentum().data();
    for (std::size_t i = 0; i < nflat; ++i) {
        m[i] = -m[i];
    }

    hmc.integrate_only(0.3, 10);

    double max_u_err        = 0.0;
    double max_p_err        = 0.0;
    double const* const upf = u.data();
    double const* const ppf = hmc.momentum().data();
    for (std::size_t i = 0; i < nflat; ++i) {
        max_u_err = std::max(max_u_err, std::abs(upf[i] - u0[i]));
        max_p_err = std::max(max_p_err, std::abs(ppf[i] - (-p0[i])));
    }
    REQUIRE(max_u_err < 1e-9);
    REQUIRE(max_p_err < 1e-9);
}

}  // namespace

TEST_CASE("Wilson<SU3> HMC Leapfrog is reversible to ~1e-9",
          "[physics][gauge][su3][hmc][reversibility]") {
    check_reversibility<Leapfrog>();
}

TEST_CASE("Wilson<SU3> HMC Omelyan2 is reversible to ~1e-9",
          "[physics][gauge][su3][hmc][reversibility]") {
    check_reversibility<Omelyan2>();
}

TEST_CASE("Wilson<SU3> HMC Omelyan4 is reversible to ~1e-9",
          "[physics][gauge][su3][hmc][reversibility]") {
    check_reversibility<Omelyan4>();
}
