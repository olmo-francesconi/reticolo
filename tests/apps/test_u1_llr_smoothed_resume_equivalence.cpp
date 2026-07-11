// LLR checkpoint / --resume regression for the smoothed driver (compact U(1)
// Wilson). Complements the standard-driver tests: this exercises
// Orchestrator::run_smoothed's resume path (NR skip, RM restart, per-sweep snapshot).
// The cross-replica smoother carries no state beyond each replica's a, so the
// same ensemble snapshot suffices; the resume tail must be bit-identical.

#include "llr_resume_helpers.hpp"

#include <string>

#include <catch2/catch_test_macros.hpp>

#ifndef U1_LLR_SMOOTHED_BINARY
    #error "U1_LLR_SMOOTHED_BINARY compile definition is required"
#endif

TEST_CASE("u1_llr_smoothed --resume reproduces the continuation bit-exact (smoothed driver)",
          "[app][e2e][u1_llr_smoothed][llr][checkpoint][regression]") {
    // 3 replicas (E 200..600 step 200); warm-up runs on the fresh legs, is
    // skipped on resume (fields come from the checkpoint).
    std::string const phys_args =
        " -L 4 --ndim=4 --E_min=200 --E_max=600 --delta=200"
        " --n_therm_nr=2 --n_meas_nr=4 --n_therm_rm=2 --n_meas_rm=4 --n_md=8 --seed=20260710"
        " --replica_threads=1";
    reticolo::test::require_llr_resume_equivalence(
        U1_LLR_SMOOTHED_BINARY, phys_args, /*n_nr=*/2, /*n_rm=*/6, /*ckpt_sweep=*/3, "u1sm_m1");
}
