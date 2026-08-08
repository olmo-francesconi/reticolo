#include <reticolo/action/complex.hpp>
#include <reticolo/action/gauge.hpp>
#include <reticolo/action/nn.hpp>
#include <reticolo/core/rng/fast_rng.hpp>
#include <reticolo/updater/metropolis/metropolis.hpp>

#include <cmath>
#include <complex>
#include <cstddef>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

// The load-bearing test for every local-update path: the ΔS a sweep accumulates
// must equal the change in the action, measured independently.
//
// `metropolis_stencil` scores a move from a LOCAL energy — the leaf's own kernel on
// the all-directions aggregate for the scalar families, the staple factorisation
// Re Tr[U·A] for gauge. Each of those is an identity that holds only if every
// bond touching the moved element is counted exactly once. If a family gets that
// counting wrong the sweep still runs, still accepts at a plausible rate, and
// still produces configurations — it just samples the wrong distribution. The
// only thing that catches it is comparing the carried S against a full `s_full`
// recomputed from scratch.
//
// A handful of sweeps on a small lattice is enough: the identity is exact per
// move, so an error shows up immediately rather than needing statistics. These
// are consistency checks, not Monte Carlo runs.

namespace {

using Catch::Matchers::WithinRel;

constexpr int k_sweeps = 6;

// Run a few sweeps, then compare the incrementally carried S against a fresh
// s_full. `tol` is relative; f64 paths are exact to rounding.
template <class Action, class Field>
void require_carried_action_matches(Action const& a, Field& f, double sigma, double tol) {
    reticolo::updater::Metropolis m{
        a, f, reticolo::FastRng{4242}, {.sigma = sigma}, reticolo::log::Mode::silent};

    std::size_t n_attempts = 0;
    std::size_t n_accepted = 0;
    for (int i = 0; i < k_sweeps; ++i) {
        auto const st = m.step(reticolo::log::Mode::silent);
        n_attempts += st.n_attempts;
        n_accepted += st.n_accepted;
    }

    // A sweep that accepts nothing (or everything) would pass the comparison
    // below trivially — a frozen field has ΔS = 0 on both sides. Require the
    // chain to have actually moved before trusting the identity.
    REQUIRE(n_attempts > 0);
    REQUIRE(n_accepted > 0);
    REQUIRE(n_accepted < n_attempts);

    REQUIRE_THAT(m.last_s_full(), WithinRel(a.s_full(f), tol));
}

}  // namespace

TEST_CASE("NN scalar local energy matches s_full", "[physics][metropolis][nn]") {
    using namespace reticolo;

    SECTION("Phi4 3D — site leaf, identity combine") {
        Lattice<double> phi{{8, 8, 8}};
        action::Phi4<double> const a{.kappa = 0.12, .lambda = 0.5};
        require_carried_action_matches(a, phi, 0.4, 1e-12);
    }
    SECTION("Phi4 4D") {
        Lattice<double> phi{{6, 6, 6, 6}};
        action::Phi4<double> const a{.kappa = 0.18, .lambda = 1.1};
        require_carried_action_matches(a, phi, 0.4, 1e-12);
    }
    SECTION("Phi6 3D") {
        Lattice<double> phi{{8, 8, 8}};
        action::Phi6<double> const a{.kappa = 0.10, .lambda = 0.4, .g6 = 0.2};
        require_carried_action_matches(a, phi, 0.35, 1e-12);
    }
    SECTION("Xy 3D — BOND leaf, whose combine reads the candidate") {
        // The case that forces the per-candidate fold: a bond combine is a
        // function of both endpoints, so the two candidate values genuinely
        // aggregate to different sums. A single shared fold passes Phi4 and
        // fails here.
        Lattice<double> theta{{8, 8, 8}};
        action::Xy<double> const a{.beta = 0.9};
        require_carried_action_matches(a, theta, 0.5, 1e-12);
    }
    SECTION("SineGordon 3D — scalar-cos kernel, not the batched s_full") {
        Lattice<double> phi{{8, 8, 8}};
        action::SineGordon<double> const a{.kappa = 0.10, .alpha = 0.7};
        require_carried_action_matches(a, phi, 0.4, 1e-12);
    }
    SECTION("Phi4 f32 — the per-site energy is evaluated at f32") {
        // Each ΔS is a difference of two f32 energies, so it carries ~1e-5
        // absolute error and the carried S drifts. Harmless for the accept test;
        // the tolerance here records the real behaviour rather than hiding it.
        Lattice<float> phi{{8, 8, 8}};
        action::Phi4<float> const a{.kappa = 0.12F, .lambda = 0.5F};
        require_carried_action_matches(a, phi, 0.4, 1e-5);
    }
}

TEST_CASE("Complex local energy matches s_full", "[physics][metropolis][complex]") {
    using namespace reticolo;
    using Field = Lattice<std::complex<double>>;

    // The identity survives the anisotropy because Re(conj(a)·b) is symmetric, so
    // a bond contributes equally to either endpoint. Sweeping mu (including a
    // negative one) is what would expose a forward/backward asymmetry leaking
    // from S_I into the sampled weight.
    SECTION("BoseGas mu = 0") {
        Field phi{{8, 8, 8}};
        action::BoseGas<double> const a{.mass = 1.0, .lambda = 1.0, .mu = 0.0};
        require_carried_action_matches(a, phi, 0.3, 1e-12);
    }
    SECTION("BoseGas mu = 1.5") {
        Field phi{{6, 6, 6, 6}};
        action::BoseGas<double> const a{.mass = 0.7, .lambda = 0.5, .mu = 1.5};
        require_carried_action_matches(a, phi, 0.25, 1e-12);
    }
    SECTION("BoseGas mu = -0.9") {
        Field phi{{12, 12}};
        action::BoseGas<double> const a{.mass = 1.2, .lambda = 2.0, .mu = -0.9};
        require_carried_action_matches(a, phi, 0.3, 1e-12);
    }
}

TEMPLATE_TEST_CASE("Wilson local energy matches s_full",
                   "[physics][metropolis][gauge]",
                   reticolo::math::group::U1,
                   reticolo::math::group::SU2,
                   reticolo::math::group::SU3) {
    using namespace reticolo;
    using Action = action::Wilson<TestType, double>;
    using Field  = typename Action::field_type;

    // beta is chosen per group so the sweep neither freezes nor accepts
    // everything at sigma = 0.3 (require_carried_action_matches asserts both).
    double const beta = (TestType::n_color == 1) ? 1.0 : (TestType::n_color == 2 ? 2.0 : 5.5);

    SECTION("3D") {
        Action const a{.beta = beta};
        Field u{{6, 6, 6}};
        set_cold_identity(u);  // a default link field is ZERO-filled, not identity
        require_carried_action_matches(a, u, 0.3, 1e-12);
    }
    SECTION("4D") {
        Action const a{.beta = beta};
        Field u{{4, 4, 4, 4}};
        set_cold_identity(u);
        require_carried_action_matches(a, u, 0.3, 1e-12);
    }
}
