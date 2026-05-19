// Plaquette match for SU(2) Wilson HMC at deep weak coupling β = 8 on 4D L=4.
//
// Literature reference: the leading-order weak-coupling expansion of the
// Wilson plaquette is
//     ⟨P⟩ → 1 − (N²−1)/(d·β)       as β → ∞
// — derived in every lattice-gauge-theory textbook (Creutz, "Quarks, Gluons
// and Lattices" Ch. 6; Gattringer & Lang, "QCD on the Lattice" §4.4) by
// expanding U_μ(x) = exp(iA_μ) to quadratic order and applying equipartition
// over the gauge-boson modes. For SU(2) 4D this gives ⟨P⟩ = 1 − 3/(4β),
// which at β = 8 is 0.90625. Corrections are O(1/β²) ~ 0.015 at this β;
// finite-volume corrections at L = 4 are well below that for plaquettes
// (an UV-dominated observable).
//
// Test tolerance 0.005 absorbs the LO correction + statistical noise
// (~0.002 on 200 trajectories) with comfortable margin.

#include <reticolo/action/detail/gauge_group/su2.hpp>
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
using reticolo::gauge_group::SU2;

TEST_CASE("Wilson<SU2> 4D L=4 β=8 matches LO weak-coupling plaquette (Creutz/Gattringer-Lang)",
          "[physics][gauge][su2][hmc][weak-coupling]") {
    constexpr double k_beta             = 8.0;
    constexpr std::size_t k_l           = 4;
    constexpr std::size_t k_ndim        = 4;
    constexpr int k_n_therm             = 200;
    constexpr int k_n_prod              = 200;
    constexpr double k_tau              = 1.0;
    constexpr int k_n_md                = 20;
    // Literature: ⟨P⟩(∞) ≈ 1 − (N²−1)/(d·β) = 1 − 3/(4·8) = 0.90625
    constexpr double k_lit_plaq         = 1.0 - (3.0 / (4.0 * k_beta));
    constexpr double k_tol              = 0.005;

    Wilson<SU2, double> const action{.beta = k_beta};
    MatrixLinkLattice<SU2, double>::SizeVec const shape(k_ndim, k_l);
    MatrixLinkLattice<SU2, double> links{shape};
    std::size_t const ns = links.nsites();
    for (std::size_t mu = 0; mu < k_ndim; ++mu) {
        double* const blk = links.mu_block_data(mu);
        for (std::size_t s = 0; s < ns; ++s) {
            blk[(0 * ns) + s] = 1.0;
            blk[(6 * ns) + s] = 1.0;
        }
    }
    FastRng rng{271828};
    Hmc<Wilson<SU2, double>, FastRng, Omelyan2, MatrixLinkLattice<SU2, double>> hmc{
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
