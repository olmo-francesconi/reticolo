#include "smoke_helpers.hpp"

#ifndef PHI4_LLR_METROPOLIS_BINARY
    #error "PHI4_LLR_METROPOLIS_BINARY compile definition is required"
#endif

// Same schema assertions as the HMC LLR apps — that is the point. The sampler is
// a template parameter of the orchestrator, so a Metropolis LLR run must produce
// a byte-compatible /cfg + /replica_NNN + /exchange layout; only the CLI knob
// changes (sigma for tau/n_md).
TEST_CASE("phi4_llr_metropolis binary writes the expected HDF5 schema",
          "[app][e2e][phi4_llr_metropolis]") {
    reticolo::test::require_llr_smoke({.binary  = PHI4_LLR_METROPOLIS_BINARY,
                                       .tag     = "phi4_llr_metropolis_smoke",
                                       .physics = "--kappa=0.18 --lambda=1.0",
                                       .window  = "--E_min=0 --E_max=20 --delta=10",
                                       .sampler = "--sigma=0.5"});
}
