#include "smoke_helpers.hpp"

#ifndef PHI4_HMC_BINARY
    #error "PHI4_HMC_BINARY compile definition is required"
#endif

// End-to-end: run the real binary on a tiny lattice and check it wrote the
// documented HDF5 schema. Harness in smoke_helpers.hpp; this file is the table
// entry — binary, couplings, and the observables phi4_hmc writes.
TEST_CASE("phi4_hmc binary writes the expected HDF5 schema", "[app][e2e][phi4_hmc]") {
    reticolo::test::require_hmc_smoke({.binary  = PHI4_HMC_BINARY,
                                       .tag     = "phi4_smoke",
                                       .physics = "--kappa=0.13 --lambda=0.02",
                                       .obs     = {"s", "mag", "m2"}});
}
