#include "smoke_helpers.hpp"

#ifndef SU3_METROPOLIS_BINARY
    #error "SU3_METROPOLIS_BINARY compile definition is required"
#endif

// The link-field entry: a sweep here is 2*ndim colour passes, not 2, so this
// also exercises the (direction, parity) colouring end to end.
TEST_CASE("su3_metropolis binary writes the expected HDF5 schema", "[app][e2e][su3_metropolis]") {
    reticolo::test::require_metropolis_smoke({.binary  = SU3_METROPOLIS_BINARY,
                                              .tag     = "su3_metropolis_smoke",
                                              .physics = "--beta=5.7 --ndim=3",
                                              .obs     = {"s", "plaq"},
                                              .sigma   = 0.3});
}
