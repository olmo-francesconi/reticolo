#include <reticolo/action/detail/concepts.hpp>
#include <reticolo/action/on_sigma.hpp>
#include <reticolo/algorithm/metropolis.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <array>
#include <cmath>
#include <cstddef>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using reticolo::FastRng;
using reticolo::Lattice;
using reticolo::Site;
using reticolo::action::HasProposal;
using reticolo::action::HasSEff;
using reticolo::action::LocalAction;
using reticolo::action::OnSigma;
using reticolo::alg::Metropolis;

using O3      = OnSigma<3>;
using O3Field = std::array<double, 3>;

static_assert(LocalAction<O3, O3Field>);
static_assert(HasSEff<O3, O3Field>);
static_assert(HasProposal<O3, O3Field, FastRng>);

namespace {

void seed_aligned(Lattice<O3Field>& phi) {
    O3Field const ex{1.0, 0.0, 0.0};
    for (Site const x : phi.sites()) {
        phi[x] = ex;
    }
}

double norm(O3Field const& v) {
    return std::sqrt((v[0] * v[0]) + (v[1] * v[1]) + (v[2] * v[2]));
}

}  // namespace

TEST_CASE("OnSigma::propose returns unit vectors", "[physics][on_sigma]") {
    O3 const action{.beta = 0.5};
    Lattice<O3Field> phi{{2, 2}};
    seed_aligned(phi);
    FastRng rng{42};
    for (int trial = 0; trial < 256; ++trial) {
        auto const proposal = action.propose(phi, Site{0}, rng);
        REQUIRE(norm(proposal) == Catch::Approx(1.0).margin(1e-12));
    }
}

TEST_CASE("OnSigma: ds_local matches s_full difference", "[physics][on_sigma]") {
    O3 const action{.beta = 0.7};
    Lattice<O3Field> phi{{4, 4}};
    seed_aligned(phi);
    FastRng rng{17};
    // Randomise (preserving the unit-norm constraint).
    for (Site const x : phi.sites()) {
        phi[x] = action.propose(phi, x, rng);
    }

    for (std::size_t trial = 0; trial < 25; ++trial) {
        Site const x      = Site{rng.uniform_int(phi.nsites())};
        O3Field const old = phi[x];
        O3Field const nv  = action.propose(phi, x, rng);

        double const ds_predicted = action.ds_local(phi, x, nv);
        double const s_old        = action.s_full(phi);
        phi[x]                    = nv;
        double const s_new        = action.s_full(phi);
        phi[x]                    = old;

        REQUIRE(ds_predicted == Catch::Approx((s_new - s_old)).margin(1e-10));
    }
}

TEST_CASE("OnSigma at beta=0: Metropolis reaches isotropic distribution",
          "[physics][on_sigma][metropolis]") {
    O3 const action{.beta = 0.0};
    Lattice<O3Field> phi{{6, 6}};
    seed_aligned(phi);
    FastRng rng{12345};

    Metropolis<O3, FastRng, O3Field> mc{
        action, phi, rng, reticolo::alg::MetropolisSpec{.sigma = 1.0}};

    // beta=0 means every proposal is accepted; one sweep already randomises.
    for (int s = 0; s < 20; ++s) {
        auto const stats = mc.step();
        REQUIRE(stats.accepted == stats.attempts);
    }

    constexpr int n_meas = 2000;
    std::array<double, 3> sum_v{};
    std::array<double, 3> sum_v2{};
    std::size_t samples = 0;
    for (int meas = 0; meas < n_meas; ++meas) {
        (void)mc.step();
        for (Site const x : phi.sites()) {
            for (std::size_t i = 0; i < 3; ++i) {
                sum_v[i] += phi[x][i];
                sum_v2[i] += phi[x][i] * phi[x][i];
            }
            ++samples;
        }
    }

    auto const inv_n = 1.0 / static_cast<double>(samples);
    // Each component <phi_i> = 0 (isotropy) and <phi_i^2> = 1/N = 1/3.
    for (std::size_t i = 0; i < 3; ++i) {
        double const mean    = sum_v[i] * inv_n;
        double const mean_sq = sum_v2[i] * inv_n;
        REQUIRE(mean == Catch::Approx(0.0).margin(0.02));
        REQUIRE(mean_sq == Catch::Approx((1.0 / 3.0)).margin(0.02));
    }
}

TEST_CASE("OnSigma at large beta: NN dot product approaches 1", "[physics][on_sigma][metropolis]") {
    O3 const action{.beta = 4.0};
    Lattice<O3Field> phi{{6, 6}};
    seed_aligned(phi);
    FastRng rng{99};

    Metropolis<O3, FastRng, O3Field> mc{
        action, phi, rng, reticolo::alg::MetropolisSpec{.sigma = 1.0}};
    for (int s = 0; s < 800; ++s) {
        (void)mc.step();
    }

    constexpr int n_meas   = 400;
    double sum_dot         = 0.0;
    std::size_t bond_count = 0;
    for (int meas = 0; meas < n_meas; ++meas) {
        (void)mc.step();
        for (Site const x : phi.sites()) {
            for (std::size_t mu = 0; mu < phi.ndims(); ++mu) {
                sum_dot += O3::dot(phi[x], phi[phi.next(x, mu)]);
                ++bond_count;
            }
        }
    }
    double const mean_dot = sum_dot / static_cast<double>(bond_count);
    INFO("<phi(x).phi(y)> = " << mean_dot << " (expect ~1 at beta=4)");
    REQUIRE(mean_dot > 0.7);
}
