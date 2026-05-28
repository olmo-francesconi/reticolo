#include <reticolo/action/detail/gauge_group/su2.hpp>
#include <reticolo/action/wilson.hpp>
#include <reticolo/algorithm/hmc.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/rng.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::MatrixLinkLattice;
using reticolo::action::Wilson;
using reticolo::alg::Hmc;
using reticolo::alg::HmcSpec;
using SU2 = reticolo::gauge_group::SU2;

// The Wilson S-reduction returns double whatever the link precision — the
// mixed-precision invariant (the β·n_plaq − (β/N)·Σ combine would cancel
// catastrophically in float otherwise).
static_assert(std::is_same_v<decltype(std::declval<Wilson<SU2, float> const&>().s_full(
                                 std::declval<MatrixLinkLattice<SU2, float> const&>())),
                             double>);
static_assert(std::is_same_v<decltype(std::declval<Wilson<SU2, double> const&>().s_full(
                                 std::declval<MatrixLinkLattice<SU2, double> const&>())),
                             double>);

namespace {

template <class T>
void cold_start(MatrixLinkLattice<SU2, T>& u) {
    std::size_t const ns = u.nsites();
    for (std::size_t mu = 0; mu < u.ndims(); ++mu) {
        T* const blk = u.mu_block_data(mu);
        for (std::size_t s = 0; s < ns; ++s) {
            blk[(0 * ns) + s] = T{1};
            blk[(6 * ns) + s] = T{1};
        }
    }
}

// Mean plaquette ⟨P⟩ = 1 − ⟨S⟩/(β·n_plaq) from a short HMC run; also reports
// acceptance.
template <class T>
double equilibrium_plaq(double beta, unsigned long long seed, double& acc_out) {
    MatrixLinkLattice<SU2, T> u{{4, 4, 4, 4}};
    cold_start(u);
    FastRng rng{seed};
    Wilson<SU2, T> const action{.beta = static_cast<T>(beta)};
    Hmc hmc{action, u, rng, HmcSpec{.tau = 1.0, .n_md = 20}, reticolo::alg::integ::omelyan2};

    for (int i = 0; i < 200; ++i) {
        (void)hmc.step(reticolo::log::Mode::silent);
    }
    std::size_t const ns     = u.nsites();
    std::size_t const n_plaq = (4U * 3U / 2U) * ns;
    double const norm        = beta * static_cast<double>(n_plaq);

    constexpr int n_meas = 400;
    double sum_s         = 0.0;
    long accepted        = 0;
    for (int m = 0; m < n_meas; ++m) {
        accepted += hmc.step(reticolo::log::Mode::silent).accepted ? 1 : 0;
        sum_s += action.s_full(u);
    }
    acc_out = static_cast<double>(accepted) / static_cast<double>(n_meas);
    return 1.0 - ((sum_s / static_cast<double>(n_meas)) / norm);
}

}  // namespace

// Float and double SU(2) HMC must agree on the mean plaquette. The float
// trajectory integrates the links, momenta and staple force in single
// precision, but ΔH is double-accumulated so the chain targets exp(-S).
TEST_CASE("SU(2) HMC: float and double agree on the mean plaquette",
          "[physics][su2][hmc][mixed-precision]") {
    constexpr double k_beta = 2.3;

    double acc_d        = 0.0;
    double acc_f        = 0.0;
    double const plaq_d = equilibrium_plaq<double>(k_beta, 2024, acc_d);
    double const plaq_f = equilibrium_plaq<float>(k_beta, 4048, acc_f);

    INFO("<plaq> double = " << plaq_d << "  (acc " << acc_d << ")");
    INFO("<plaq> float  = " << plaq_f << "  (acc " << acc_f << ")");

    REQUIRE(acc_f > 0.5);
    REQUIRE(plaq_d > 0.0);
    REQUIRE(plaq_d < 1.0);
    REQUIRE(std::abs(plaq_f - plaq_d) < 0.01);
}
