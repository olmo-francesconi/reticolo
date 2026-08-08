#include "smoke_helpers.hpp"

#ifndef SINE_GORDON_METROPOLIS_BINARY
    #error "SINE_GORDON_METROPOLIS_BINARY compile definition is required"
#endif

// Table entry: binary, couplings, and the observables sine_gordon_metropolis writes.
TEST_CASE("sine_gordon_metropolis binary writes the expected HDF5 schema",
          "[app][e2e][sine_gordon_metropolis]") {
    reticolo::test::require_metropolis_smoke({.binary  = SINE_GORDON_METROPOLIS_BINARY,
                                              .tag     = "sine_gordon_metropolis_smoke",
                                              .physics = "--kappa=0.12 --alpha=0.5",
                                              .obs     = {"s", "mag", "cos_phi"},
                                              .sigma   = 0.3});
}
