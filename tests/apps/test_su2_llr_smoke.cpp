#include "smoke_helpers.hpp"

#ifndef SU2_LLR_BINARY
    #error "SU2_LLR_BINARY compile definition is required"
#endif

TEST_CASE("su2_llr binary writes the expected HDF5 schema", "[app][e2e][su2_llr]") {
    reticolo::test::require_llr_smoke({.binary  = SU2_LLR_BINARY,
                                       .tag     = "su2_llr_smoke",
                                       .physics = "--beta=2.0",
                                       .window  = "--E_min=0 --E_max=60 --delta=20"});
}
