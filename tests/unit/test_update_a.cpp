#include <reticolo/orch/llr/ramp.hpp>
#include <reticolo/orch/llr/update_a.hpp>

#include <cmath>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinRel;
using reticolo::orch::llr::nr_update;
using reticolo::orch::llr::ramp_centre;
using reticolo::orch::llr::ramp_steps;
using reticolo::orch::llr::rm_update;

// The window coefficient is C = 1, NOT C = 12. The Gaussian penalty
// (E - E_n)²/(2δ²) has variance δ², so the Newton step is <dE>/δ²; the quoted
// 12/δ² belongs to the HARD window (indicator of width δ, variance δ²/12).
// Mixing them makes the step 12x too aggressive and the slopes diverge
// geometrically — a past incident in this tree, and until now the only thing
// preventing its recurrence was a comment in update_a.hpp.
TEST_CASE("nr_update is the Gaussian-window Newton step a + <dE>/delta^2", "[llr]") {
    REQUIRE_THAT(nr_update(0.0, 1.0, 1.0), WithinRel(1.0, 1e-15));
    REQUIRE_THAT(nr_update(0.5, 2.0, 2.0), WithinRel(1.0, 1e-15));
    REQUIRE_THAT(nr_update(-1.0, 0.25, 0.5), WithinRel(0.0, 1e-15));

    // The hard-window value would be 12x this step. Pin the gap explicitly so a
    // coefficient swap fails here rather than in a diverging production run.
    double const step = nr_update(0.0, 1.0, 2.0);
    REQUIRE_THAT(step, WithinRel(0.25, 1e-15));
    REQUIRE(step * 12.0 != step);
}

// Robbins-Monro is the same step damped by 1/(k+1); k restarts at 0 after the
// NR warm-up, so k = 0 must reproduce the undamped Newton step exactly.
TEST_CASE("rm_update damps the Newton step by 1/(k+1)", "[llr]") {
    REQUIRE_THAT(rm_update(0.0, 1.0, 1.0, 0), WithinRel(nr_update(0.0, 1.0, 1.0), 1e-15));

    double const a = 0.3;
    for (int k : {0, 1, 3, 9}) {
        double const damped = rm_update(a, 1.0, 2.0, k) - a;
        double const full   = nr_update(a, 1.0, 2.0) - a;
        INFO("k=" << k);
        REQUIRE_THAT(damped * static_cast<double>(k + 1), WithinRel(full, 1e-15));
    }
}

// Both updates are pure in `a`: a zero mean deviation is a fixed point, and the
// step is odd in <dE> (the sign of the correction follows which side of the
// window centre the replica sits on).
TEST_CASE("a zero mean deviation leaves the slope unchanged", "[llr]") {
    REQUIRE_THAT(nr_update(0.7, 0.0, 1.5), WithinRel(0.7, 1e-15));
    REQUIRE_THAT(rm_update(0.7, 0.0, 1.5, 4), WithinRel(0.7, 1e-15));

    REQUIRE_THAT(nr_update(0.0, -1.0, 2.0), WithinRel(-nr_update(0.0, 1.0, 2.0), 1e-15));
    REQUIRE_THAT(rm_update(0.0, -1.0, 2.0, 3), WithinRel(-rm_update(0.0, 1.0, 2.0, 3), 1e-15));
}

// Float instantiation: the LLR ladder is templated on the replica scalar, so the
// updates must compile and behave for float as well as double.
TEST_CASE("update_a is scalar-generic", "[llr]") {
    REQUIRE_THAT(nr_update(0.0F, 1.0F, 2.0F), WithinRel(0.25F, 1e-6F));
    REQUIRE_THAT(rm_update(0.0F, 1.0F, 2.0F, 1), WithinRel(0.125F, 1e-6F));
}

// The deep-window ramp schedule (orch/llr/ramp.hpp) is shared by the CPU and
// CUDA warm-in drivers — a §2.4 category-3 quantity: not per-site math, but its
// absence is the difference between a seated replica and a NaN trajectory. It
// went missing on the CUDA side once already because each backend wrote its own
// warm loop.
TEST_CASE("ramp: no intermediate centres when the field already sits in the window",
          "[llr][ramp]") {
    // |gap| <= delta/2 needs no walk at all.
    REQUIRE(ramp_steps(0.0, 1.0) == 0);
    REQUIRE(ramp_steps(0.4, 1.0) == 0);
    // Degenerate delta must not divide by zero or loop forever.
    REQUIRE(ramp_steps(5.0, 0.0) == 0);
    REQUIRE(ramp_steps(5.0, -1.0) == 0);
}

// Every hop is at most half a window width — the bound that keeps the window
// force (Q - E_n)/delta^2 inside the integrator's stable step. A field
// equilibrated at the old centre has spread ~delta, so moving the centre by
// half a width leaves it well inside the new window.
TEST_CASE("ramp: every step is at most half a window width", "[llr][ramp]") {
    for (double gap : {1.0, 2.0, 5.0, 37.5, -12.25}) {
        for (double delta : {0.5, 2.0, 7.0}) {
            int const n = ramp_steps(gap, delta);
            INFO("gap=" << gap << " delta=" << delta << " n=" << n);
            double prev = 0.0;  // q0 = 0
            for (int k = 1; k <= n; ++k) {
                double const c = ramp_centre(0.0, gap, k, n);
                REQUIRE(std::abs(c - prev) <= (0.5 * delta) + 1e-12);
                prev = c;
            }
            if (n > 0) {
                REQUIRE_THAT(ramp_centre(0.0, gap, n, n), WithinRel(gap, 1e-15));
            }
        }
    }
}

// Symmetric in direction, and offset by q0 rather than assuming a zero start.
TEST_CASE("ramp: centres are anchored at q0 and signed by the gap", "[llr][ramp]") {
    REQUIRE(ramp_steps(-8.0, 1.0) == ramp_steps(8.0, 1.0));
    REQUIRE_THAT(ramp_centre(3.0, 4.0, 2, 4), WithinRel(5.0, 1e-15));
    REQUIRE_THAT(ramp_centre(3.0, -4.0, 2, 4), WithinRel(1.0, 1e-15));
}
