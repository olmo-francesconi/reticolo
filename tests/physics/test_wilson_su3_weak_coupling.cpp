// Plaquette match for SU(3) Wilson HMC at the canonical lattice-QCD β = 6.0
// on 4D L=4.
//
// Literature reference: the SU(3) Wilson plaquette ⟨P⟩ = ⟨(1/3) Re Tr U_p⟩
// at β = 6.0 in the infinite-volume limit is one of the most-measured
// quantities in lattice gauge theory. The canonical value
//     ⟨P⟩(∞)|_{β=6.0} ≈ 0.5937
// is established to 4-decimal precision across many independent groups —
// see e.g. Bali & Schilling, "Static quark-antiquark potential: scaling
// behaviour and finite-size effects in SU(3) lattice gauge theory", Phys.
// Rev. D 47, 661 (1993), and follow-up high-precision measurements (e.g.
// Edwards, Heller, Klassen 1998).
//
// β = 6.0 is the canonical QCD-relevant Wilson coupling: small enough that
// the leading-order Gaussian-saddle formula (N²−1)/(d·β) = 0.333 misses
// the true value by ~5%, so this test cannot use the LO formula and must
// anchor on the published non-perturbative value. We use a 0.01 tolerance
// on |⟨P⟩ − 0.5937|, which absorbs:
//   - L=4 finite-volume correction from L=∞ (~0.004 on this lattice)
//   - statistical noise on 200 trajectories (~0.002)
//   - small differences from the precise scheme/extrapolation used in the
//     reference.

#include <reticolo/action/detail/gauge_group/su3.hpp>
#include <reticolo/action/wilson.hpp>
#include <reticolo/algorithm/hmc.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/rng.hpp>

#include <cmath>
#include <cstddef>

#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::MatrixLinkLattice;
using reticolo::action::Wilson;
using reticolo::alg::Hmc;
using reticolo::alg::integ::Omelyan2;
using reticolo::gauge_group::SU3;

TEST_CASE("Wilson<SU3> 4D L=4 β=6.0 matches Bali-Schilling 1993 mean plaquette",
          "[physics][gauge][su3][hmc][weak-coupling]") {
    constexpr double k_beta             = 6.0;
    constexpr std::size_t k_l           = 4;
    constexpr std::size_t k_ndim        = 4;
    constexpr int k_n_therm             = 200;
    constexpr int k_n_prod              = 200;
    constexpr double k_tau              = 1.0;
    constexpr int k_n_md                = 30;
    constexpr double k_lit_plaq         = 0.5937;
    constexpr double k_tol              = 0.01;

    Wilson<SU3, double> const action{.beta = k_beta};
    MatrixLinkLattice<SU3, double>::SizeVec const shape(k_ndim, k_l);
    MatrixLinkLattice<SU3, double> links{shape};
    std::size_t const ns = links.nsites();
    for (std::size_t mu = 0; mu < k_ndim; ++mu) {
        double* const blk = links.mu_block_data(mu);
        for (std::size_t s = 0; s < ns; ++s) {
            blk[(0 * ns) + s]  = 1.0;
            blk[(8 * ns) + s]  = 1.0;
            blk[(16 * ns) + s] = 1.0;
        }
    }
    FastRng rng{271828};
    Hmc<Wilson<SU3, double>, FastRng, Omelyan2, MatrixLinkLattice<SU3, double>> hmc{
        action, links, rng, {.tau = k_tau, .n_md = k_n_md}};

    for (int i = 0; i < k_n_therm; ++i) {
        (void)hmc.trajectory();
    }

    double const n_plaq    = static_cast<double>(k_ndim * (k_ndim - 1) / 2 * ns);
    double const plaq_norm = k_beta * n_plaq;
    double plaq_sum        = 0.0;
    for (int i = 0; i < k_n_prod; ++i) {
        (void)hmc.trajectory();
        plaq_sum += 1.0 - (action.s_full(links) / plaq_norm);
    }
    double const plaq_mean = plaq_sum / static_cast<double>(k_n_prod);
    REQUIRE(std::abs(plaq_mean - k_lit_plaq) < k_tol);
}
