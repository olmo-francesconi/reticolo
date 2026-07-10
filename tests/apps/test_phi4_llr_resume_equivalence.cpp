// End-to-end regression test for the LLR ensemble checkpoint / --resume path
// (standard driver, phi4 scalar action). A baseline run and a checkpoint+resume
// run must produce a bit-identical RM tail — pinning per-replica field + HMC
// StreamSet + a, plus the orchestrator exch_rng and schedule position. Covered
// at m=1 (serial replicas, the default) and m=2 (threaded HMC per replica) so a
// regression in either the serial or the nested-threading RNG path diverges.

#include "llr_resume_helpers.hpp"

#include <string>

#include <catch2/catch_test_macros.hpp>

#ifndef PHI4_LLR_BINARY
    #error "PHI4_LLR_BINARY compile definition is required"
#endif

namespace {

// 3 replicas (E 0..50 step 25), tiny lattice + trajectory counts: a smoke-sized
// deterministic chain, not a physics run.
std::string phys_args(int replica_threads) {
    return " -L 4 --ndim=2 --E_min=0 --E_max=50 --delta=25"
           " --n_therm_nr=2 --n_meas_nr=4 --n_therm_rm=2 --n_meas_rm=4 --n_md=10 --seed=20260710"
           " --replica_threads=" +
           std::to_string(replica_threads);
}

}  // namespace

TEST_CASE("phi4_llr --resume reproduces the continuation bit-exact (m=1, serial replicas)",
          "[app][e2e][phi4_llr][llr][checkpoint][regression]") {
    reticolo::test::require_llr_resume_equivalence(
        PHI4_LLR_BINARY, phys_args(1), /*n_nr=*/2, /*n_rm=*/6, /*ckpt_sweep=*/3, "phi4_llr_m1");
}

TEST_CASE("phi4_llr --resume reproduces the continuation bit-exact (m=2, threaded replicas)",
          "[app][e2e][phi4_llr][llr][checkpoint][threads][regression]") {
    reticolo::test::require_llr_resume_equivalence(PHI4_LLR_BINARY,
                                                   phys_args(2),
                                                   /*n_nr=*/2,
                                                   /*n_rm=*/6,
                                                   /*ckpt_sweep=*/3,
                                                   "phi4_llr_m2",
                                                   /*omp_threads=*/4);
}
