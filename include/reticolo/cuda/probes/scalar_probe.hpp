#pragma once

namespace reticolo::cuda {

// Phase 3 gates (defined in src/cuda/scalar_probe.cu): the generic DeviceAction
// over each new scalar action's functor pair reproduces the CPU action's s_full
// and force — the shared HD formula is one source of truth across CPU + device.
// Polynomial actions match to roundoff; the transcendental ones (SineGordon, XY)
// match to a bounded tolerance (device intrinsic vs Sleef/libm, ~1 ULP/site).

[[nodiscard]] bool phi6_cpu_matches_device();
[[nodiscard]] bool sine_gordon_cpu_matches_device();
[[nodiscard]] bool xy_cpu_matches_device();

}  // namespace reticolo::cuda
