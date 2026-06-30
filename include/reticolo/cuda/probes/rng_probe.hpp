#pragma once

namespace reticolo::cuda {

// RNG gates (defined in src/cuda/rng_probe.cu).

// The device Philox uniforms are BIT-IDENTICAL to the host philox_uniform2 for
// the same (seed, traj, index) — the cross-backend reproducibility proof
// (Philox is integer-only; the uniform scaling is exact).
[[nodiscard]] bool philox_host_matches_device();

// Advancing the trajectory counter changes the sampled momenta; the same
// counter reproduces them exactly. Guards the graph-capture baked-counter trap.
[[nodiscard]] bool philox_traj_distinct();

// A large device fill has mean ≈ 0, variance ≈ 1 (deterministic seed, so this
// is a single correctness check, not a flaky statistical trial).
[[nodiscard]] bool philox_moments_ok();

}  // namespace reticolo::cuda
