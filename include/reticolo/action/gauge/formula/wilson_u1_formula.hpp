#pragma once

#include <reticolo/core/hd.hpp>

// Shared Wilson-U(1) plaquette angle — the one piece of per-plaquette math common
// to the CPU host kernels (wilson_kernels<U1>) and the CUDA device kernels
// (gauge_u1.cuh). Only the ANGLE is shared: the loop STRUCTURE deliberately
// differs (the host scatters plane-by-plane with Sleef-batched cos/sin; the
// device gathers per-link/per-site with intrinsic cos/sin, honouring the
// gather-only device invariant). cos/sin are applied by each side in its own loop.

namespace reticolo::action::formula {

// Oriented plaquette angle theta_{mu,nu}(x) from its four link angles:
//   +theta_mu(x) + theta_nu(x+mu) - theta_mu(x+nu) - theta_nu(x)
template <class T>
[[nodiscard]] RETICOLO_HD inline T u1_plaq(T a_mu_x, T b_nu_xpmu, T c_mu_xpnu, T d_nu_x) {
    return a_mu_x + b_nu_xpmu - c_mu_xpnu - d_nu_x;
}

}  // namespace reticolo::action::formula
