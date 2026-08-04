#include "smoke_helpers.hpp"

#ifndef U1_HMC_BINARY
    #error "U1_HMC_BINARY compile definition is required"
#endif

TEST_CASE("u1_hmc binary writes the expected HDF5 schema", "[app][e2e][u1_hmc]") {
    reticolo::test::require_hmc_smoke({.binary  = U1_HMC_BINARY,
                                       .tag     = "u1_hmc_smoke",
                                       .physics = "--ndim=2 --beta=1.0 --tau=1.0 --n_md=10",
                                       .obs     = {"s", "plaq"},
                                       .n_therm = 5,
                                       .n_prod  = 15,
                                       .seed    = 20260518});
}
