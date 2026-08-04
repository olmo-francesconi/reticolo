#include "smoke_helpers.hpp"

#ifndef BOSE_GAS_LLR_BINARY
    #error "BOSE_GAS_LLR_BINARY compile definition is required"
#endif

// Complex LLR: the window constrains the imaginary observable S_I (mode B).
// Same replica schema as the real-action LLR apps.
TEST_CASE("bose_gas_llr binary writes the expected HDF5 schema", "[app][e2e][bose_gas_llr]") {
    reticolo::test::require_llr_smoke({.binary  = BOSE_GAS_LLR_BINARY,
                                       .tag     = "bose_gas_llr_smoke",
                                       .physics = "--mass=1.0 --lambda=1.0 --mu=0.5",
                                       .window  = "--E_min=-6 --E_max=6 --delta=3",
                                       .seed    = 20260701});
}
