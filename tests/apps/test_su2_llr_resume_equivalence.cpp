// LLR checkpoint / --resume regression for the SU(2) Wilson gauge action
// (standard driver, MatrixLinkLattice). Complements the phi4 scalar test: this
// exercises the gauge-field checkpoint path (link buffer + per-slab StreamSet
// over a matrix-link field). m=1; the resume tail must be bit-identical.

#include "llr_resume_helpers.hpp"

#include <string>

#include <catch2/catch_test_macros.hpp>

#ifndef SU2_LLR_BINARY
    #error "SU2_LLR_BINARY compile definition is required"
#endif

TEST_CASE("su2_llr --resume reproduces the continuation bit-exact (gauge action)",
          "[app][e2e][su2_llr][llr][checkpoint][regression]") {
    // 3 replicas (E 200..600 step 200); cold-started, no warm-up phase.
    std::string const phys_args =
        " -L 4 --ndim=4 --E_min=200 --E_max=600 --delta=200"
        " --n_therm_nr=2 --n_meas_nr=4 --n_therm_rm=2 --n_meas_rm=4 --n_md=8 --seed=20260710"
        " --replica_threads=1";
    reticolo::test::require_llr_resume_equivalence(
        SU2_LLR_BINARY, phys_args, /*n_nr=*/2, /*n_rm=*/6, /*ckpt_sweep=*/3, "su2_llr_m1");
}
