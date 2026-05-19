#include <reticolo/action/detail/concepts.hpp>
#include <reticolo/action/phi4.hpp>
#include <reticolo/action/sine_gordon.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <cmath>
#include <cstddef>

#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::Lattice;
using reticolo::Site;
using reticolo::action::HasForce;
using reticolo::action::HasSEff;
using reticolo::action::LocalAction;
using reticolo::action::Phi4;
using reticolo::action::SineGordon;

static_assert(LocalAction<SineGordon<double>, double>);
static_assert(HasSEff<SineGordon<double>, double>);
static_assert(HasForce<SineGordon<double>, double>);

namespace {

void randomize(Lattice<double>& phi, FastRng& rng) {
    for (Site const x : phi.sites()) {
        phi[x] = 0.5 * rng.normal();
    }
}

}  // namespace

TEST_CASE("SineGordon: ds_local matches finite difference of s_full", "[physics][sine_gordon]") {
    SineGordon<double> const action{.kappa = 0.15, .alpha = 0.3};

    Lattice<double> phi{{6, 6, 6}};
    FastRng rng{1729};
    randomize(phi, rng);

    for (std::size_t trial = 0; trial < 20; ++trial) {
        Site const x     = Site{rng.uniform_int(phi.nsites())};
        double const old = phi[x];
        double const nv  = old + (0.4 * rng.normal());

        double const ds_predicted = action.ds_local(phi, x, nv);
        double const s_old        = action.s_full(phi);
        phi[x]                    = nv;
        double const s_new        = action.s_full(phi);
        phi[x]                    = old;

        REQUIRE(std::abs(ds_predicted - (s_new - s_old)) < 1e-9);
    }
}

TEST_CASE("SineGordon: compute_force matches central FD of s_full", "[physics][sine_gordon]") {
    SineGordon<double> const action{.kappa = 0.17, .alpha = 0.4};

    Lattice<double> phi{{6, 6, 6}};
    Lattice<double> force{phi.indexing()};
    FastRng rng{2718};
    randomize(phi, rng);

    action.compute_force(phi, force);

    constexpr double k_eps = 1e-4;
    constexpr double k_tol = 1e-6;

    for (std::size_t trial = 0; trial < 25; ++trial) {
        Site const x     = Site{rng.uniform_int(phi.nsites())};
        double const old = phi[x];

        phi[x]               = old + k_eps;
        double const s_plus  = action.s_full(phi);
        phi[x]               = old - k_eps;
        double const s_minus = action.s_full(phi);
        phi[x]               = old;

        double const grad_numeric = (s_plus - s_minus) / (2.0 * k_eps);
        REQUIRE(std::abs(force[x] - (-grad_numeric)) < k_tol);
    }
}

TEST_CASE("SineGordon at alpha=0 reduces to Phi4 with lambda=0", "[physics][sine_gordon]") {
    SineGordon<double> const sg{.kappa = 0.18, .alpha = 0.0};
    Phi4<double> const p4{.kappa = 0.18, .lambda = 0.0};

    Lattice<double> phi{{4, 4, 4}};
    FastRng rng{42};
    randomize(phi, rng);

    REQUIRE(std::abs(sg.s_full(phi) - p4.s_full(phi)) < 1e-12);

    Lattice<double> f_sg{phi.indexing()};
    Lattice<double> f_p4{phi.indexing()};
    sg.compute_force(phi, f_sg);
    p4.compute_force(phi, f_p4);
    for (Site const x : phi.sites()) {
        REQUIRE(std::abs(f_sg[x] - f_p4[x]) < 1e-12);
    }
}
