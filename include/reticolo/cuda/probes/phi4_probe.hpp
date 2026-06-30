#pragma once

namespace reticolo::cuda {

// Phase 2a gates (defined in src/cuda/phi4_probe.cu).

// A Lattice<double> survives Lattice → DeviceField → Lattice unchanged.
[[nodiscard]] bool phi4_roundtrip_ok();

// The device DeviceAction<action::Phi4> reproduces the CPU action::Phi4
// s_full and compute_force to roundoff on a fixed config — the single
// source-of-truth proof (both call the shared HD per-site formula).
[[nodiscard]] bool phi4_cpu_matches_device();

}  // namespace reticolo::cuda
