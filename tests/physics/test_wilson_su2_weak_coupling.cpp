// Weak-coupling sanity check for SU(2) Wilson HMC at β = 8 on 4D L=4.
// Theory: ⟨P⟩ ≡ ⟨(1/N) Re Tr U_p⟩ ≈ 1 − 3/(4β) at leading order; corrections
// scale as 1/β². At β=8 the LO prediction is 0.906; we require agreement to
// within 1% (~10σ on a 600-trajectory measurement).

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

TEST_CASE("Wilson<SU2> 4D L=4 β=8 recovers weak-coupling mean plaquette",
          "[physics][gauge][su2][hmc][weak-coupling]") {
    constexpr double k_beta             = 8.0;
    constexpr std::size_t k_l           = 4;
    constexpr std::size_t k_ndim        = 4;
    constexpr int k_n_therm             = 300;
    constexpr int k_n_prod              = 600;
    constexpr double k_tau              = 1.0;
    constexpr int k_n_md                = 20;

    Wilson<SU2, double> const action{.beta = k_beta};
    MatrixLinkLattice<SU2, double>::SizeVec const shape(k_ndim, k_l);
    MatrixLinkLattice<SU2, double> links{shape};
    // Cold start: identity on every link.
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
    double const plaq_mean      = plaq_sum / static_cast<double>(k_n_prod);
    double const plaq_theory_lo = 1.0 - (3.0 / (4.0 * k_beta));
    REQUIRE(std::abs(plaq_mean - plaq_theory_lo) < 0.01);
}
