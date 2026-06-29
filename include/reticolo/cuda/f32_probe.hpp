#pragma once

namespace reticolo::cuda {

// Phase 3d gates (defined in src/cuda/f32_probe.cu): the device HMC stack
// instantiated in single precision. f32 fields run the MD drift/kick in float
// (axpy_f32) while the Hamiltonian reductions still accumulate in double, so the
// checks use a BOUNDED tolerance, never roundoff (and f32 never gets the
// integrator-order-scaling test — the dt floor is above the f32 noise).

// DeviceAction<Phi4<float>> reproduces the CPU action::Phi4<float> s_full + force
// to f32-bounded tolerance — the shared formula runs in single precision on both.
[[nodiscard]] bool phi4_f32_cpu_matches_device();

// Leapfrog MD on DeviceField<float> is time-reversible to an f32-bounded
// tolerance (run forward, negate momenta, run forward → field returns to start).
[[nodiscard]] bool hmc_f32_reversibility_ok();

}  // namespace reticolo::cuda
