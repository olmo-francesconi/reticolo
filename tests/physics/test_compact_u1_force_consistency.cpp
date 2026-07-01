#include <reticolo/action/detail/concepts.hpp>
#include <reticolo/action/gauge/compact_u1.hpp>
#include <reticolo/core/link_lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <cmath>
#include <cstddef>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::LinkLattice;
using reticolo::Site;
using reticolo::action::CompactU1;
using reticolo::action::HasFusedKick;
using reticolo::action::HmcAction;

static_assert(HmcAction<CompactU1<double>, LinkLattice<double>>);
static_assert(HasFusedKick<CompactU1<double>, LinkLattice<double>>);

template <class T>
static void randomize(LinkLattice<T>& l, FastRng& rng) {
    T* const d          = l.data();
    std::size_t const n = l.nlinks();
    for (std::size_t i = 0; i < n; ++i) {
        d[i] = static_cast<T>(rng.normal());
    }
}

TEST_CASE("CompactU1: compute_force matches central finite difference of s_full", "[physics][u1]") {
    CompactU1<double> const action{.beta = 1.0};
    LinkLattice<double> links{{4, 4, 4, 4}};
    LinkLattice<double> force{links.indexing()};
    FastRng rng{56789};
    randomize(links, rng);

    action.compute_force(links, force);

    constexpr double k_eps = 1e-4;
    constexpr double k_tol = 1e-7;

    for (std::size_t trial = 0; trial < 25; ++trial) {
        Site const x     = Site{rng.uniform_int(links.nsites())};
        std::size_t mu   = rng.uniform_int(links.ndims());
        double const old = links(x, mu);

        links(x, mu)        = old + k_eps;
        double const s_plus = action.s_full(links);

        links(x, mu)         = old - k_eps;
        double const s_minus = action.s_full(links);

        links(x, mu) = old;

        // Standard convention: compute_force returns the force F = -dS/dtheta.
        double const grad_numeric    = (s_plus - s_minus) / (2.0 * k_eps);
        double const force_predicted = force(x, mu);
        double const force_numeric   = -grad_numeric;

        REQUIRE(force_predicted == Catch::Approx(force_numeric).margin(k_tol));
    }
}

TEST_CASE("CompactU1: compute_force_and_kick matches compute_force + manual kick",
          "[physics][u1]") {
    CompactU1<double> const action{.beta = 1.5};
    LinkLattice<double> links{{4, 4, 4, 4}};
    LinkLattice<double> mom_a{links.indexing(), 0.7};
    LinkLattice<double> mom_b{links.indexing(), 0.7};
    LinkLattice<double> force{links.indexing()};
    FastRng rng{11};
    randomize(links, rng);

    constexpr double k_dt = 0.123;

    action.compute_force(links, force);
    double* const ma       = mom_a.data();
    double const* const fp = force.data();
    for (std::size_t i = 0; i < mom_a.nlinks(); ++i) {
        ma[i] += k_dt * fp[i];
    }

    action.compute_force_and_kick(links, mom_b, k_dt);

    double const* const mb = mom_b.data();
    for (std::size_t i = 0; i < mom_a.nlinks(); ++i) {
        REQUIRE(ma[i] == Catch::Approx(mb[i]).margin(1e-12));
    }
}

TEST_CASE("CompactU1: aligned plaquette config (all theta=0) has S = 0", "[physics][u1]") {
    // Standard Wilson convention: S = beta * sum (1 - cos theta_p). With all
    // theta=0, every plaquette contributes 0.
    constexpr double k_beta = 2.7;
    CompactU1<double> const action{.beta = k_beta};
    LinkLattice<double> links{{6, 6, 6}, 0.0};
    REQUIRE(action.s_full(links) == Catch::Approx(0.0).margin(1e-9));
}
