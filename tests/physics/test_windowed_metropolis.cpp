#include <reticolo/action/complex.hpp>
#include <reticolo/action/concepts.hpp>
#include <reticolo/action/nn.hpp>
#include <reticolo/action/windowed_action.hpp>
#include <reticolo/core/field/lattice.hpp>
#include <reticolo/core/log/log.hpp>
#include <reticolo/core/rng/fast_rng.hpp>
#include <reticolo/updater/metropolis/metropolis.hpp>

#include <complex>
#include <cstddef>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

// The local (Metropolis) sweep through a WindowedAction.
//
// The window is what makes this its own case rather than another action leaf:
// S_win = S_base + a·Q + (Q − E_n)²/2δ² is quadratic in a GLOBAL scalar, so the
// per-site ΔS depends on the current Q and the checkerboard's independence
// argument fails. The sweep is therefore ONE sequential pass carrying Q as a
// running scalar. What has to hold is that the running scalars stay exact — a
// drift there does not merely mis-report, it feeds straight back into the accept
// test of every subsequent site.
//
// No chain is run to equilibrium here; these are per-sweep identities, checked
// on a handful of sweeps from a cold start (see the project rule on simulations
// in tests). The HMC-vs-Metropolis distributional agreement lives outside the
// suite.

namespace {

using reticolo::Lattice;

// A second NN action used as an arbitrary constraint observable, to exercise the
// path where Q is NOT the base action.
template <class T>
struct Quadratic : reticolo::action::NNAction<Quadratic<T>, T> {
    using value_type = T;
    T c              = T{1};
    [[nodiscard]] auto action_kernel() const {
        return [c = c](T self, T /*agg*/) { return c * self * self; };
    }
    [[nodiscard]] auto force_kernel() const {
        return [c = c](std::size_t, T self, T) { return T{-2} * c * self; };
    }
};

}  // namespace

TEST_CASE("a windowed action sweeps as one sequential colour", "[physics][llr][metropolis]") {
    using namespace reticolo;
    using Windowed = action::WindowedAction<action::Phi4<double>>;

    static_assert(action::LocalAction<Windowed, Lattice<double>>,
                  "a real NN base with the self constraint must support the local sweep");

    Lattice<double> l{{8, 8}};
    Windowed const w{.base = {.kappa = 0.15, .lambda = 0.5}, .a = -1.0, .E_n = 20.0, .delta = 5.0};
    REQUIRE(w.n_colors(l) == 1);
}

TEST_CASE("the windowed sweep carries S_win exactly", "[physics][llr][metropolis]") {
    using namespace reticolo;
    using Windowed = action::WindowedAction<action::Phi4<double>>;

    Lattice<double> l{{8, 8}};
    Windowed const w{.base = {.kappa = 0.15, .lambda = 0.5}, .a = -1.0, .E_n = 20.0, .delta = 5.0};

    updater::Metropolis m{w, l, FastRng{7}, {.sigma = 0.4}, log::Mode::silent};
    double running = m.last_s_full();
    for (int i = 0; i < 8; ++i) {
        auto const st = m.step(log::Mode::silent);
        REQUIRE(st.n_attempts == l.nsites());  // one colour, every site attempted
        REQUIRE(st.n_accepted > 0);            // a frozen chain would pass everything below
        REQUIRE(st.n_accepted < st.n_attempts);
        running += st.ds;
    }
    // Σ of the accepted ΔS_win must reproduce a fresh windowed total.
    REQUIRE(running == Catch::Approx(w.s_full(l)).epsilon(1e-11));
    REQUIRE(m.last_s_full() == Catch::Approx(w.s_full(l)).epsilon(1e-11));
}

TEST_CASE("the windowed sweep publishes the caches LLR reads", "[physics][llr][metropolis]") {
    using namespace reticolo;

    // orch::llr::Replica never re-derives S_base or Q — it reads the caches the
    // sampler leaves behind, for ⟨dE⟩, for exchange and for the tilt update. HMC
    // populates them at the end of a trajectory; the local sweep must leave them
    // in the same state, from its running scalars rather than from a fresh pass.
    SECTION("self constraint: Q is the base action") {
        using Windowed = action::WindowedAction<action::Phi4<double>>;
        Lattice<double> l{{8, 8}};
        Windowed const w{
            .base = {.kappa = 0.15, .lambda = 0.5}, .a = -1.0, .E_n = 20.0, .delta = 5.0};
        updater::Metropolis m{w, l, FastRng{9}, {.sigma = 0.4}, log::Mode::silent};
        for (int i = 0; i < 8; ++i) {
            (void)m.step(log::Mode::silent);
        }
        REQUIRE(w.last_s_full() == Catch::Approx(w.base.s_full(l)).epsilon(1e-11));
        REQUIRE(w.last_constraint() == Catch::Approx(w.constraint_value(l)).epsilon(1e-11));
    }

    SECTION("observable constraint: Q is a separate action") {
        using C        = action::ObservableConstraint<Quadratic<double>>;
        using Windowed = action::WindowedAction<action::Phi4<double>, C>;
        Lattice<double> l{{8, 8}};
        Windowed const w{.base       = {.kappa = 0.15, .lambda = 0.5},
                         .a          = 0.2,
                         .E_n        = 30.0,
                         .delta      = 5.0,
                         .constraint = C{.obs = Quadratic<double>{}}};
        updater::Metropolis m{w, l, FastRng{9}, {.sigma = 0.4}, log::Mode::silent};
        for (int i = 0; i < 8; ++i) {
            (void)m.step(log::Mode::silent);
        }
        // The two caches are genuinely different quantities on this path.
        REQUIRE(w.last_s_full() == Catch::Approx(w.base.s_full(l)).epsilon(1e-11));
        REQUIRE(w.last_constraint() == Catch::Approx(w.constraint_value(l)).epsilon(1e-11));
        REQUIRE(w.last_s_full() != Catch::Approx(w.last_constraint()));
    }
}

TEST_CASE("a windowed sweep accepts an odd lattice extent", "[physics][llr][metropolis]") {
    using namespace reticolo;
    using Windowed = action::WindowedAction<action::Phi4<double>>;

    // The even-extent rule belongs to the checkerboard, not to Metropolis: a
    // single-colour sequential pass never assumes the two parities are disjoint,
    // so the updater must not refuse the lattice on its behalf.
    Lattice<double> odd{{8, 7}};
    Windowed const w{.base = {.kappa = 0.15, .lambda = 0.5}, .a = -1.0, .E_n = 20.0, .delta = 5.0};
    updater::Metropolis m{w, odd, FastRng{3}, {.sigma = 0.4}, log::Mode::silent};
    auto const st = m.step(log::Mode::silent);
    REQUIRE(st.n_attempts == odd.nsites());
}

TEST_CASE("a sign-problem window has no local sweep yet", "[physics][llr][metropolis]") {
    using namespace reticolo;
    using Field    = Lattice<std::complex<double>>;
    using Windowed = action::WindowedAction<action::BoseGas<double>>;

    // ImagConstraint has no local form for Q = S_I, so the windowed action must
    // FAIL action::LocalAction rather than satisfy it and then fail to compile
    // (or, worse, sample the wrong measure) at the first sweep. The guard on
    // WindowedAction::metropolis_stencil is what makes this a concept check.
    static_assert(!action::LocalAction<Windowed, Field>,
                  "the imaginary-part constraint must not claim a local sweep");
    static_assert(action::HmcAction<Windowed, Field>, "…while HMC keeps working");
}
