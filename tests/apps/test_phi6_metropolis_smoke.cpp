#include "smoke_helpers.hpp"

#ifndef PHI6_METROPOLIS_BINARY
    #error "PHI6_METROPOLIS_BINARY compile definition is required"
#endif

// Table entry: binary, couplings, and the observables phi6_metropolis writes.
TEST_CASE("phi6_metropolis binary writes the expected HDF5 schema", "[app][e2e][phi6_metropolis]") {
    reticolo::test::require_metropolis_smoke({.binary  = PHI6_METROPOLIS_BINARY,
                                              .tag     = "phi6_metropolis_smoke",
                                              .physics = "--kappa=0.10 --lambda=0.02 --g6=0.01",
                                              .obs     = {"s", "mag", "m2"},
                                              .sigma   = 0.3});
}
