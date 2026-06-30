#include <reticolo/action/gauge/compact_u1.hpp>
#include <reticolo/action/site/phi4.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/link_lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <cmath>
#include <cstddef>
#include <numbers>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::flat_size;
using reticolo::Lattice;
using reticolo::LinkLattice;
using reticolo::Site;
using reticolo::action::CompactU1;
using reticolo::action::Phi4;

// `s_full_and_force` must reproduce what the separate `s_full` +
// `compute_force` kernels compute: the force exactly (same per-site
// expression and traversal), the action up to reassociation (the fused
// kernel rebuilds it from the full neighbour sum / a sincos pass).

TEST_CASE("Phi4: s_full_and_force matches s_full + compute_force", "[unit][phi4][fused]") {
    Phi4<double> const action{.kappa = 0.18, .lambda = 1.0};

    Lattice<double> phi{{6, 6, 6}};
    Lattice<double> f_fused{phi.indexing()};
    Lattice<double> f_ref{phi.indexing()};
    FastRng rng{1234};
    for (Site x : phi.sites()) {
        phi[x] = rng.normal();
    }

    double const s_ref   = action.s_full(phi);
    double const s_fused = action.s_full_and_force(phi, f_fused);
    action.compute_force(phi, f_ref);

    REQUIRE(s_fused == Catch::Approx(s_ref).epsilon(1e-12));
    for (Site x : phi.sites()) {
        REQUIRE(f_fused[x] == Catch::Approx(f_ref[x]).margin(1e-13));
    }
}

TEST_CASE("Phi4: s_full_and_force leaves the last_s_full cache untouched", "[unit][phi4][fused]") {
    Phi4<double> const action{.kappa = 0.13, .lambda = 0.05};

    Lattice<double> phi{{4, 4, 4}};
    Lattice<double> force{phi.indexing()};
    FastRng rng{77};
    for (Site x : phi.sites()) {
        phi[x] = rng.normal();
    }

    double const cached = action.s_full(phi);
    phi[Site{0}] += 1.0;
    (void)action.s_full_and_force(phi, force);

    REQUIRE(action.last_s_full() == cached);
}

TEST_CASE("CompactU1: s_full_and_force matches s_full + compute_force",
          "[unit][compact_u1][fused]") {
    CompactU1<double> const action{.beta = 1.0};

    LinkLattice<double> u{{4, 4, 4, 4}};
    LinkLattice<double> f_fused{u.indexing()};
    LinkLattice<double> f_ref{u.indexing()};
    FastRng rng{4321};
    for (auto& link : u) {
        link = (rng.uniform() - 0.5) * 2.0 * std::numbers::pi;
    }

    double const s_ref   = action.s_full(u);
    double const s_fused = action.s_full_and_force(u, f_fused);
    action.compute_force(u, f_ref);

    REQUIRE(s_fused == Catch::Approx(s_ref).epsilon(1e-12));
    double const* fp = f_fused.data();
    double const* rp = f_ref.data();
    for (std::size_t i = 0; i < flat_size(f_fused); ++i) {
        REQUIRE(fp[i] == Catch::Approx(rp[i]).margin(1e-13));
    }
}

TEST_CASE("CompactU1<float>: s_full_and_force matches s_full + compute_force",
          "[unit][compact_u1][fused]") {
    CompactU1<float> const action{.beta = 1.0F};

    LinkLattice<float> u{{4, 4, 4, 4}};
    LinkLattice<float> f_fused{u.indexing()};
    LinkLattice<float> f_ref{u.indexing()};
    FastRng rng{999};
    for (auto& link : u) {
        link = static_cast<float>((rng.uniform() - 0.5) * 2.0 * std::numbers::pi);
    }

    double const s_ref   = action.s_full(u);
    double const s_fused = action.s_full_and_force(u, f_fused);
    action.compute_force(u, f_ref);

    REQUIRE(s_fused == Catch::Approx(s_ref).epsilon(1e-4));
    float const* fp = f_fused.data();
    float const* rp = f_ref.data();
    for (std::size_t i = 0; i < flat_size(f_fused); ++i) {
        REQUIRE(fp[i] == Catch::Approx(rp[i]).margin(1e-5));
    }
}
