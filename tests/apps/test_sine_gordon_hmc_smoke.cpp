#include "smoke_helpers.hpp"

#ifndef SINE_GORDON_HMC_BINARY
    #error "SINE_GORDON_HMC_BINARY compile definition is required"
#endif

// End-to-end: run the real binary on a tiny lattice and check it wrote the
// documented HDF5 schema. Harness in smoke_helpers.hpp; this file is the table
// entry — binary, couplings, and the observables sine_gordon_hmc writes.
TEST_CASE("sine_gordon_hmc binary writes the expected HDF5 schema", "[app][e2e][sine_gordon_hmc]") {
    reticolo::test::require_hmc_smoke({.binary  = SINE_GORDON_HMC_BINARY,
                                       .tag     = "sine_gordon_smoke",
                                       .physics = "--kappa=0.15 --alpha=0.3",
                                       .obs     = {"s", "mag", "cos_phi"}});
}
