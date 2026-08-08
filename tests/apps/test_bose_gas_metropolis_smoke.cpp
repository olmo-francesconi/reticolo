#include "smoke_helpers.hpp"

#ifndef BOSE_GAS_METROPOLIS_BINARY
    #error "BOSE_GAS_METROPOLIS_BINARY compile definition is required"
#endif

// The complex entry: the proposal is a complex Gaussian and the sampled weight is
// the phase-quenched S_R, while S_I is measured alongside it.
TEST_CASE("bose_gas_metropolis binary writes the expected HDF5 schema",
          "[app][e2e][bose_gas_metropolis]") {
    reticolo::test::require_metropolis_smoke(
        {.binary  = BOSE_GAS_METROPOLIS_BINARY,
         .tag     = "bose_gas_metropolis_smoke",
         .physics = "--mass=1.0 --lambda=1.0 --mu=0.5 --ndim=3",
         .obs     = {"s_r", "s_i"},
         .sigma   = 0.3});
}
