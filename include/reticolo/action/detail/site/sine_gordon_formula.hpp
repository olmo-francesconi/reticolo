#pragma once

#include <reticolo/core/hd.hpp>

// Shared Sine-Gordon per-site formulas — single source of truth for the CPU
// action (action::SineGordon, sine_gordon.hpp) and the CUDA device functors.
// The transcendental is passed IN (sin_phi / cos_phi) rather than computed
// here, so the algebra is identical across paths while each chooses its own
// evaluator: Sleef-batched on the CPU f64 hot loop, a device intrinsic on the
// GPU, scalar std::sin/std::cos on the CPU non-double fallback. The values
// match to ~1 ULP — the same cross-evaluator transcendental caveat the Philox
// normals already carry.

namespace reticolo::action::detail {

// F(x) = 2k·nbrs - 2phi - alpha·sin(phi), given the sum over ALL 2*ndims
// neighbours and the precomputed sin(phi).
template <class T>
[[nodiscard]] RETICOLO_HD inline T
sine_gordon_force_site(T phi, T nbrs, T sin_phi, T kappa, T alpha) {
    return (T{2} * kappa * nbrs) - (T{2} * phi) - (alpha * sin_phi);
}

// Per-site action -2k·phi·fwd + phi^2 - alpha·cos(phi), given the sum over the
// d FORWARD neighbours and the precomputed cos(phi).
template <class T>
[[nodiscard]] RETICOLO_HD inline T
sine_gordon_action_site(T phi, T fwd, T cos_phi, T kappa, T alpha) {
    return (T{-2} * kappa * phi * fwd) + (phi * phi) - (alpha * cos_phi);
}

}  // namespace reticolo::action::detail
