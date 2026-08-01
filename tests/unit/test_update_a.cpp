#include <reticolo/orch/llr/update_a.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinRel;
using reticolo::orch::llr::nr_update;
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
