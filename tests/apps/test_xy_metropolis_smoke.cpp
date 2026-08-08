#include "smoke_helpers.hpp"

#ifndef XY_METROPOLIS_BINARY
    #error "XY_METROPOLIS_BINARY compile definition is required"
#endif

// Table entry: binary, couplings, and the observables xy_metropolis writes.
TEST_CASE("xy_metropolis binary writes the expected HDF5 schema", "[app][e2e][xy_metropolis]") {
    reticolo::test::require_metropolis_smoke({.binary  = XY_METROPOLIS_BINARY,
                                              .tag     = "xy_metropolis_smoke",
                                              .physics = "--beta=0.9 --ndim=3",
                                              .obs     = {"s", "mag"},
                                              .sigma   = 0.5});
}
