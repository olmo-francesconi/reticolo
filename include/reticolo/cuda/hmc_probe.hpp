#pragma once

namespace reticolo::cuda {

// Phase 2b gates (defined in src/cuda/hmc_probe.cu).

// Leapfrog MD on DeviceField is time-reversible to f64 roundoff: run forward,
// negate momenta, run forward again → the field returns to its start.
[[nodiscard]] bool hmc_reversibility_ok();

// The unchanged alg::integ tags produce their nominal |ΔH| ~ dt^p scaling on
// the device: Leapfrog p≈2, Omelyan2 p≈2, Omelyan4 p≈4 (seeded short-chain
// average). The proof that the integrator seam is generic across the backend.
[[nodiscard]] bool integrator_order_ok();

// cuda::Hmc::step() wiring runs: sample → MD → ΔH → host MH accept/restore over
// several steps with finite ΔH and a field that stays finite.
[[nodiscard]] bool hmc_step_runs();

// Phase 2d: a graph-captured MD trajectory (and its replay) reproduces the
// eager MD field bit-for-bit from the same (q0, p0).
[[nodiscard]] bool graph_replay_matches_eager();

// Phase 2e: K host-free trajectories (graph replay + device-side Metropolis
// accept, no per-step sync) are deterministic — two runs from the same seed
// agree bit-for-bit — with a sane acceptance and a finite field.
[[nodiscard]] bool hmc_device_run_deterministic();

}  // namespace reticolo::cuda
