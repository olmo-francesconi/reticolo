#pragma once

namespace reticolo::cuda {

// Phase 1 exit gate. Runs the scalar device protocol end-to-end on a dummy
// Phi4-shaped functor pair: computes the MD force with the stencil skeleton
// and the total action with the reduce_fwd skeleton, then verifies
// force[j] == -dS/dphi(j) by central finite differences at a spread of sites.
// Returns true iff every probed site matches to FD tolerance. Defined in
// src/cuda/stencil_probe.cu.
[[nodiscard]] bool stencil_force_matches_fd();

}  // namespace reticolo::cuda
