#include "smoke_helpers.hpp"

#ifndef PHI6_HMC_BINARY
    #error "PHI6_HMC_BINARY compile definition is required"
#endif

// End-to-end: run the real binary on a tiny lattice and check it wrote the
// documented HDF5 schema. Harness in smoke_helpers.hpp; this file is the table
// entry — binary, couplings, and the observables phi6_hmc writes.
TEST_CASE("phi6_hmc binary writes the expected HDF5 schema", "[app][e2e][phi6_hmc]") {
    reticolo::test::require_hmc_smoke({.binary  = PHI6_HMC_BINARY,
                                       .tag     = "phi6_smoke",
                                       .physics = "--kappa=0.13 --lambda=0.02 --g6=0.01",
                                       .obs     = {"s", "mag", "m2"}});
}
