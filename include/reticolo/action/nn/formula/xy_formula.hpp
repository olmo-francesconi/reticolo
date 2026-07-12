#pragma once

#include <reticolo/core/sys/hd.hpp>

#include <cmath>

// Shared XY (planar rotor) per-bond formulas — single source of truth for the
// CPU action (action::Xy, xy.hpp) and the CUDA device functors. Each bond
// contributes a transcendental of the angle DIFFERENCE; the site finalize scales
// the accumulated sum by -beta. std::sin/std::cos are __host__ __device__ under
// nvcc (a device intrinsic) and plain libm on the host — the same per-bond math
// the CPU action already inlined.

namespace reticolo::action::formula {

// Force bond: contribution of one neighbour to -dS/dtheta(x) before the -beta.
template <class T>
[[nodiscard]] RETICOLO_HD inline T xy_force_bond(T theta, T nbr) {
    return std::sin(theta - nbr);
}

// Action bond: contribution of one neighbour to S(x) before the -beta.
template <class T>
[[nodiscard]] RETICOLO_HD inline T xy_action_bond(T theta, T nbr) {
    return std::cos(theta - nbr);
}

}  // namespace reticolo::action::formula
