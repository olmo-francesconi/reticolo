#include <reticolo/action/on_sigma.hpp>
#include <reticolo/action/xy.hpp>
#include <reticolo/action/detail/concepts.hpp>
#include <reticolo/algorithm/wolff.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>

#include <catch2/catch_test_macros.hpp>

using reticolo::FastRng;
using reticolo::Lattice;
using reticolo::Site;
using reticolo::action::OnSigma;
using reticolo::action::WolffEmbeddable;
using reticolo::action::Xy;
using reticolo::alg::Wolff;

static_assert(WolffEmbeddable<Xy<double>, double, FastRng>);
static_assert(WolffEmbeddable<OnSigma<3>, std::array<double, 3>, FastRng>);

// ---- Reflection involution: Wolff reflect must satisfy R^2 = id. ------------

TEST_CASE("Xy: wolff_reflect is an involution", "[physics][wolff][xy]") {
    Xy<double> const action{.beta = 0.5};
    FastRng rng{1};
    for (int trial = 0; trial < 50; ++trial) {
        double const axis  = action.wolff_random_axis(rng);
        double const theta = 2.0 * std::numbers::pi * rng.uniform();
        double const back  = action.wolff_reflect(action.wolff_reflect(theta, axis), axis);
        REQUIRE(std::abs(std::cos(back) - std::cos(theta)) < 1e-12);
        REQUIRE(std::abs(std::sin(back) - std::sin(theta)) < 1e-12);
    }
}

TEST_CASE("OnSigma<3>: wolff_reflect is an involution", "[physics][wolff][on]") {
    OnSigma<3> const action{.beta = 0.5};
    FastRng rng{2};
    for (int trial = 0; trial < 50; ++trial) {
        auto const axis = action.wolff_random_axis(rng);
        auto phi        = action.wolff_random_axis(rng);  // also unit
        auto const back = action.wolff_reflect(action.wolff_reflect(phi, axis), axis);
        for (std::size_t i = 0; i < 3; ++i) {
            REQUIRE(std::abs(back[i] - phi[i]) < 1e-12);
        }
    }
}

// ---- Single-cluster invariants ---------------------------------------------
//
// At beta = 0 every link probability collapses to zero, so the cluster is
// always exactly the seed site. At very large beta, with an aligned starting
// configuration, the perpendicular projection (r·phi_x)(r·phi_y) is positive
// for almost every bond and the cluster grows to cover the lattice.

TEST_CASE("Xy Wolff at beta=0: cluster is always the seed alone", "[physics][wolff][xy]") {
    Xy<double> const action{.beta = 0.0};
    Lattice<double> theta{{6, 6}};
    FastRng rng{17};
    for (Site const x : theta.sites()) {
        theta[x] = 2.0 * std::numbers::pi * rng.uniform();
    }
    Wolff<Xy<double>, FastRng> w{action, theta, rng};
    for (int trial = 0; trial < 50; ++trial) {
        REQUIRE(w.update().cluster_size == 1);
    }
}

TEST_CASE("Xy Wolff at large beta: aligned start grows full-volume clusters",
          "[physics][wolff][xy]") {
    Xy<double> const action{.beta = 6.0};
    Lattice<double> theta{{8, 8}, /*fill=*/0.0};
    FastRng rng{42};
    Wolff<Xy<double>, FastRng> w{action, theta, rng};

    std::size_t big_clusters = 0;
    constexpr int n_trials   = 20;
    auto const n_sites       = theta.nsites();
    for (int trial = 0; trial < n_trials; ++trial) {
        auto const step = w.update();
        if (step.cluster_size > (n_sites * 3 / 4)) {
            ++big_clusters;
        }
    }
    INFO("big clusters = " << big_clusters << " / " << n_trials);
    REQUIRE(big_clusters >= 15);
}

// At large beta the cluster almost always engulfs the lattice: average cluster
// size should be close to the volume. (At finite beta the cluster flip does
// not in general leave s_full invariant — only frozen-bond boundary terms
// change, but those exist whenever the cluster doesn't fully engulf the
// connected component.)
TEST_CASE("OnSigma<3> Wolff at large beta: average cluster fills the volume",
          "[physics][wolff][on]") {
    OnSigma<3> const action{.beta = 6.0};
    Lattice<std::array<double, 3>> phi{{6, 6}, {1.0, 0.0, 0.0}};
    FastRng rng{7};

    Wolff<OnSigma<3>, FastRng> w{action, phi, rng};
    auto const n_sites     = phi.nsites();
    constexpr int n_trials = 50;
    std::size_t total      = 0;
    for (int trial = 0; trial < n_trials; ++trial) {
        total += w.update().cluster_size;
    }
    double const avg = static_cast<double>(total) / static_cast<double>(n_trials);
    INFO("avg cluster = " << avg << " / " << n_sites);
    REQUIRE(avg > 0.7 * static_cast<double>(n_sites));
}
