#include "smoke_helpers.hpp"

#ifndef U1_METROPOLIS_BINARY
    #error "U1_METROPOLIS_BINARY compile definition is required"
#endif

// Table entry: binary, couplings, and the observables u1_metropolis writes.
TEST_CASE("u1_metropolis binary writes the expected HDF5 schema", "[app][e2e][u1_metropolis]") {
    reticolo::test::require_metropolis_smoke({.binary  = U1_METROPOLIS_BINARY,
                                              .tag     = "u1_metropolis_smoke",
                                              .physics = "--beta=1.0 --ndim=3",
                                              .obs     = {"s", "plaq"},
                                              .sigma   = 0.4});
}
