#pragma once

#include <reticolo/core/hd.hpp>

// Shared Phi6 per-site formulas — single source of truth for the CPU action
// (action::Phi6, phi6.hpp) and the CUDA device functors. Same structure as
// phi4_formula.hpp plus the on-site g6·phi^6 term; at g6 = 0 it reduces exactly
// to the Phi4 formula. RETICOLO_HD so the identical arithmetic runs host+device.

namespace reticolo::action::formula {

// F(x) = -dS/dphi(x), given phi and the sum over ALL 2*ndims neighbours.
template <class T>
[[nodiscard]] RETICOLO_HD inline T phi6_force_site(T phi, T nbrs, T kappa, T lambda, T g6) {
    T const phi2 = phi * phi;
    T const phi5 = phi2 * phi2 * phi;
    return (T{2} * kappa * nbrs) - (T{2} * phi) - (T{4} * lambda * phi * (phi2 - T{1})) -
           (T{6} * g6 * phi5);
}

// Per-site action contribution, given phi and the sum over the d FORWARD
// neighbours (positive-mu convention, each bond counted once).
template <class T>
[[nodiscard]] RETICOLO_HD inline T phi6_action_site(T phi, T fwd, T kappa, T lambda, T g6) {
    T const phi2 = phi * phi;
    T const dev  = phi2 - T{1};
    T const phi6 = phi2 * phi2 * phi2;
    return (T{-2} * kappa * phi * fwd) + phi2 + (lambda * dev * dev) + (g6 * phi6);
}

}  // namespace reticolo::action::formula
