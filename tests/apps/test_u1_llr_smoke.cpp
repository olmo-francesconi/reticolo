#include "smoke_helpers.hpp"

#ifndef U1_LLR_BINARY
    #error "U1_LLR_BINARY compile definition is required"
#endif

TEST_CASE("u1_llr binary writes the expected HDF5 schema", "[app][e2e][u1_llr]") {
    reticolo::test::require_llr_smoke({.binary  = U1_LLR_BINARY,
                                       .tag     = "u1_llr_smoke",
                                       .physics = "--beta=1.0",
                                       .window  = "--E_min=0 --E_max=30 --delta=10"});
}
