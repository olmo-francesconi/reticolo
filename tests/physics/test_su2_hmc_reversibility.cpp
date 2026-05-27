// SU(2) Wilson HMC reversibility: integrate forward, negate the momentum,
// integrate back — the field must return to its starting configuration to
// numerical noise (~1e-10) and the momentum must equal the negated starting
// momentum. Exercises the group-aware drift (exp lmul) plus the standard
// additive kick for all three integrators.

#include <reticolo/action/detail/gauge_group/su2.hpp>
#include <reticolo/action/wilson.hpp>
#include <reticolo/algorithm/hmc.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/math/su2_ops.hpp>

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
using reticolo::gauge_group::SU2;

namespace {

// Initialize the field to per-site exp(P) with a deterministic small P, giving
// a non-trivial but valid SU(2) configuration on every link.
void hot_start(MatrixLinkLattice<SU2, double>& u, FastRng& rng) {
    std::size_t const d  = u.ndims();
    std::size_t const ns = u.nsites();
    for (std::size_t mu = 0; mu < d; ++mu) {
        double* const blk = u.mu_block_data(mu);
        // Sample algebra → exp(0.3 · P) so we get reasonable random SU(2).
        reticolo::math::su2::sample_algebra_slab(blk, rng, ns);
        reticolo::math::su2::expi_lmul_slab(blk, blk, 0.3, ns);
        // The above wrote V = exp(0.3·P)·P-as-original which is garbage; redo cleanly.
        // (sample writes P, exp_slab does U ← exp(dt·P)·U with U == P — wrong path.)
        // Use the explicit two-buffer path below instead.
    }
    // Clean approach: reset to identity, sample algebra into scratch, drift.
    std::size_t const total = u.ndims() * SU2::n_real_components * u.nsites();
    double* const data      = u.data();
    for (std::size_t i = 0; i < total; ++i) {
        data[i] = 0.0;
    }
    for (std::size_t mu = 0; mu < d; ++mu) {
        double* const blk = u.mu_block_data(mu);
        // Identity per site in this direction.
        for (std::size_t s = 0; s < ns; ++s) {
            blk[(0 * ns) + s] = 1.0;
            blk[(6 * ns) + s] = 1.0;
        }
    }
    // Now drift each direction by a random algebra element to get SU(2) noise.
    std::vector<double> scratch_p(SU2::n_real_components * ns);
    for (std::size_t mu = 0; mu < d; ++mu) {
        reticolo::math::su2::sample_algebra_slab(scratch_p.data(), rng, ns);
        reticolo::math::su2::expi_lmul_slab(u.mu_block_data(mu), scratch_p.data(), 0.5, ns);
    }
}

template <class Integrator>
void check_reversibility() {
    Wilson<SU2, double> const action{.beta = 2.4};
    MatrixLinkLattice<SU2, double> u{{4, 4, 4, 4}};
    FastRng rng{31415};
    hot_start(u, rng);

    Hmc hmc{action, u, rng, {.tau = 0.4, .n_md = 10}, Integrator{}};

    // Sample momenta into the HMC's internal buffer.
    std::size_t const d  = u.ndims();
    std::size_t const ns = u.nsites();
    for (std::size_t mu = 0; mu < d; ++mu) {
        reticolo::math::su2::sample_algebra_slab(hmc.momentum().mu_block_data(mu), rng, ns);
    }

    // Snapshot field + momentum.
    std::size_t const nflat = d * SU2::n_real_components * ns;
    std::vector<double> u0(nflat);
    std::vector<double> p0(nflat);
    double const* const up = u.data();
    double const* const pp = hmc.momentum().data();
    for (std::size_t i = 0; i < nflat; ++i) {
        u0[i] = up[i];
        p0[i] = pp[i];
    }

    hmc.integrate_only(0.4, 10);

    // Flip momentum sign in place.
    double* const m = hmc.momentum().data();
    for (std::size_t i = 0; i < nflat; ++i) {
        m[i] = -m[i];
    }

    hmc.integrate_only(0.4, 10);

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

TEST_CASE("Wilson<SU2> HMC Leapfrog is reversible to ~1e-10",
          "[physics][gauge][su2][hmc][reversibility]") {
    check_reversibility<Leapfrog>();
}

TEST_CASE("Wilson<SU2> HMC Omelyan2 is reversible to ~1e-10",
          "[physics][gauge][su2][hmc][reversibility]") {
    check_reversibility<Omelyan2>();
}

TEST_CASE("Wilson<SU2> HMC Omelyan4 is reversible to ~1e-10",
          "[physics][gauge][su2][hmc][reversibility]") {
    check_reversibility<Omelyan4>();
}
