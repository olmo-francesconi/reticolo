#pragma once

#include <reticolo/core/hd.hpp>

// Shared compact-U(1) plaquette helper — the one piece of per-plaquette math
// common to the CPU action (action::CompactU1) and the CUDA device kernels.
// Only the ANGLE is shared: the loop STRUCTURE deliberately differs (the CPU
// scatters plane-by-plane with Sleef-batched cos/sin; the device gathers
// per-link / per-site with intrinsic cos/sin), as the gather-only device
// invariant forbids reusing the CPU scatter. cos/sin are applied by each side
// in its own loop.

namespace reticolo::action::detail {

// Oriented plaquette angle theta_{mu,nu}(x) from its four link angles:
//   +theta_mu(x) + theta_nu(x+mu) - theta_mu(x+nu) - theta_nu(x)
template <class T>
[[nodiscard]] RETICOLO_HD inline T u1_plaq(T a_mu_x, T b_nu_xpmu, T c_mu_xpnu, T d_nu_x) {
    return a_mu_x + b_nu_xpmu - c_mu_xpnu - d_nu_x;
}

}  // namespace reticolo::action::detail
