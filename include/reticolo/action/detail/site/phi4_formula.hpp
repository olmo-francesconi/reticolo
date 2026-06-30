#pragma once

#include <reticolo/core/hd.hpp>

// Shared Phi4 per-site formulas — the single source of truth for both the CPU
// action (action::Phi4, phi4.hpp) and the CUDA device functors
// (cuda::Phi4ForceFunctor / Phi4EnergyFunctor). RETICOLO_HD so the exact same
// arithmetic runs on host and device; the CPU hot loops keep their own
// counter-indexed / FP-reassociated structure and only call into these for the
// scalar body, so vectorisation is unchanged.

namespace reticolo::action::detail {

// F(x) = -dS/dphi(x), given phi and the sum over ALL 2*ndims neighbours.
template <class T>
[[nodiscard]] RETICOLO_HD inline T phi4_force_site(T phi, T nbrs, T kappa, T lambda) {
    T const phi2 = phi * phi;
    return (T{2} * kappa * nbrs) - (T{2} * phi) - (T{4} * lambda * phi * (phi2 - T{1}));
}

// Per-site action contribution, given phi and the sum over the d FORWARD
// neighbours (positive-mu convention, each bond counted once).
template <class T>
[[nodiscard]] RETICOLO_HD inline T phi4_action_site(T phi, T fwd, T kappa, T lambda) {
    T const phi2 = phi * phi;
    T const dev  = phi2 - T{1};
    return (T{-2} * kappa * phi * fwd) + phi2 + (lambda * dev * dev);
}

}  // namespace reticolo::action::detail
