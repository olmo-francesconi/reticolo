// Weak-coupling sanity check for SU(3) Wilson HMC at β = 100 on 4D L=4.
// Leading-order Gaussian-saddle prediction: ⟨1 − P⟩ = (N²−1)/(d·β) = 2/β in
// 4D SU(3). At β=100 this is 0.020, with O(1/β²) corrections at the few-%
// level — we require agreement to 0.5% (~10σ on a 600-trajectory sample).

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

TEST_CASE("Wilson<SU3> 4D L=4 β=100 recovers Gaussian-saddle mean plaquette",
          "[physics][gauge][su3][hmc][weak-coupling]") {
    constexpr double k_beta      = 100.0;
    constexpr std::size_t k_l    = 4;
    constexpr std::size_t k_ndim = 4;
    constexpr int k_n_therm      = 300;
    constexpr int k_n_prod       = 600;
    constexpr double k_tau       = 1.0;
    constexpr int k_n_md         = 30;

    Wilson<SU3, double> const action{.beta = k_beta};
    MatrixLinkLattice<SU3, double>::SizeVec const shape(k_ndim, k_l);
    MatrixLinkLattice<SU3, double> links{shape};
    std::size_t const ns = links.nsites();
    for (std::size_t mu = 0; mu < k_ndim; ++mu) {
        double* const blk = links.mu_block_data(mu);
        for (std::size_t s = 0; s < ns; ++s) {
            blk[(0 * ns) + s]  = 1.0;  // Re U_{00}
            blk[(8 * ns) + s]  = 1.0;  // Re U_{11}
            blk[(16 * ns) + s] = 1.0;  // Re U_{22}
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
    double one_minus_p_sum = 0.0;
    for (int i = 0; i < k_n_prod; ++i) {
        (void)hmc.trajectory();
        one_minus_p_sum += action.s_full(links) / plaq_norm;
    }
    double const one_minus_p_mean = one_minus_p_sum / static_cast<double>(k_n_prod);
    // Saddle prediction (N²-1)/(d·β) = 8/(4·100) = 0.020.
    double const theory_lo = 8.0 / (4.0 * k_beta);
    REQUIRE(std::abs(one_minus_p_mean - theory_lo) < 0.005);
}
