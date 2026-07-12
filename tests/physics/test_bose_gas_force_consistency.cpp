#include <reticolo/action/complex/bose_gas.hpp>
#include <reticolo/action/concepts.hpp>
#include <reticolo/core/field/lattice.hpp>
#include <reticolo/core/rng/fast_rng.hpp>
#include <reticolo/core/field/site.hpp>

#include <complex>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::Lattice;
using reticolo::Site;
using reticolo::action::BoseGas;
using reticolo::action::HasFusedKick;
using reticolo::action::HasImagPart;
using reticolo::action::HmcAction;

using C = std::complex<double>;

static_assert(HmcAction<BoseGas<double>, Lattice<C>>);
static_assert(HasFusedKick<BoseGas<double>, Lattice<C>>);
static_assert(HasImagPart<BoseGas<double>, Lattice<C>>);

static void randomize(Lattice<C>& phi, FastRng& rng) {
    for (Site x : phi.sites()) {
        phi[x] = C{rng.normal(), rng.normal()};
    }
}

// F_R = -dS_R/dphi component-wise: central FD of s_full w.r.t. each of the two
// real DOFs at a site. mu != 0 exercises the cosh(mu) time-direction weight.
TEST_CASE("BoseGas: compute_force matches finite difference of s_full", "[physics][bose]") {
    BoseGas<double> const action{.mass = 1.0, .lambda = 0.5, .mu = 0.4};

    Lattice<C> phi{{4, 4, 4, 4}};
    Lattice<C> force{phi.indexing()};
    FastRng rng{56789};
    randomize(phi, rng);
    action.compute_force(phi, force);

    constexpr double k_eps = 1e-4;
    constexpr double k_tol = 1e-6;

    for (std::size_t trial = 0; trial < 25; ++trial) {
        Site const x = Site{rng.uniform_int(phi.nsites())};
        C const old  = phi[x];

        phi[x]            = old + C{k_eps, 0.0};
        double const sr_p = action.s_full(phi);
        phi[x]            = old - C{k_eps, 0.0};
        double const sr_m = action.s_full(phi);

        phi[x]            = old + C{0.0, k_eps};
        double const si_p = action.s_full(phi);
        phi[x]            = old - C{0.0, k_eps};
        double const si_m = action.s_full(phi);

        phi[x] = old;

        REQUIRE(force[x].real() == Catch::Approx(-(sr_p - sr_m) / (2.0 * k_eps)).margin(k_tol));
        REQUIRE(force[x].imag() == Catch::Approx(-(si_p - si_m) / (2.0 * k_eps)).margin(k_tol));
    }
}

// F_I = -dS_I/dphi component-wise: central FD of s_imag. S_I is mu-independent,
// so this holds for any mu.
TEST_CASE("BoseGas: compute_force_imag matches finite difference of s_imag", "[physics][bose]") {
    BoseGas<double> const action{.mass = 1.0, .lambda = 0.5, .mu = 0.4};

    Lattice<C> phi{{4, 4, 4, 4}};
    Lattice<C> force{phi.indexing()};
    FastRng rng{99};
    randomize(phi, rng);
    action.compute_force_imag(phi, force);

    constexpr double k_eps = 1e-4;
    constexpr double k_tol = 1e-6;

    for (std::size_t trial = 0; trial < 25; ++trial) {
        Site const x = Site{rng.uniform_int(phi.nsites())};
        C const old  = phi[x];

        phi[x]            = old + C{k_eps, 0.0};
        double const sr_p = action.s_imag(phi);
        phi[x]            = old - C{k_eps, 0.0};
        double const sr_m = action.s_imag(phi);

        phi[x]            = old + C{0.0, k_eps};
        double const si_p = action.s_imag(phi);
        phi[x]            = old - C{0.0, k_eps};
        double const si_m = action.s_imag(phi);

        phi[x] = old;

        REQUIRE(force[x].real() == Catch::Approx(-(sr_p - sr_m) / (2.0 * k_eps)).margin(k_tol));
        REQUIRE(force[x].imag() == Catch::Approx(-(si_p - si_m) / (2.0 * k_eps)).margin(k_tol));
    }
}
