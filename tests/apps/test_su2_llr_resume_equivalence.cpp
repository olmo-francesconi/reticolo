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
    // 3 replicas (E 200..600 step 200); cold-started. The warm-seating budget is
    // pinned small: at its 200+2000-per-replica default it costs ~10s of SU(2) 4^4
    // trajectories and dominates the whole test, while contributing nothing to
    // what is asserted — the warm phase is skipped on --resume, so it only has to
    // run identically in the full and seg1 legs, which any budget satisfies.
    std::string const phys_args =
        " -L 4 --ndim=4 --E_min=200 --E_max=600 --delta=200"
        " --n_therm_nr=2 --n_meas_nr=4 --n_therm_rm=2 --n_meas_rm=4 --n_md=8 --seed=20260710"
        " --warm_therm=4 --warm_max_traj=40 --replica_threads=1";
    reticolo::test::require_llr_resume_equivalence(
        SU2_LLR_BINARY, phys_args, /*n_nr=*/2, /*n_rm=*/6, /*ckpt_sweep=*/3, "su2_llr_m1");
}
