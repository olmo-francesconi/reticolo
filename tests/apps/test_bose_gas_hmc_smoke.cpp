#include "smoke_helpers.hpp"

#ifndef BOSE_GAS_HMC_BINARY
    #error "BOSE_GAS_HMC_BINARY compile definition is required"
#endif

// The phase-quenched Bose-gas HMC records BOTH S_R and the imaginary observable
// S_I (the mode-B LLR twin), so the schema carries /prod/obs/s_r and /prod/obs/s_i.
TEST_CASE("bose_gas_hmc binary writes S_R + S_I schema", "[app][e2e][bose_gas_hmc]") {
    reticolo::test::require_hmc_smoke(
        {.binary  = BOSE_GAS_HMC_BINARY,
         .tag     = "bose_gas_hmc_smoke",
         .physics = "--ndim=2 --mass=1.0 --lambda=1.0 --mu=0.5 --meas_every=1",
         .obs     = {"s_r", "s_i"},
         .n_therm = 8,
         .n_prod  = 12,
         .seed    = 20260701});
}
