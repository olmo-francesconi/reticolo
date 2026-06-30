#pragma once

namespace reticolo::cuda {

// Gates (defined in src/cuda/u1_probe.cu): compact U(1) gauge on the
// device. The link field is DeviceField<double, LinkLayout>; the force is the
// per-link gather (re-derived, not the CPU scatter), the action a per-site
// plaquette reduction. cuda::DeviceAction and cuda::Hmc are unchanged — the
// gauge access pattern lives only in the device_functors<CompactU1> trait.

// DeviceAction<CompactU1> reproduces the CPU action::CompactU1 s_full and force.
// s_full to roundoff; the gather force matches the CPU scatter to a bounded
// transcendental tolerance (device sin vs Sleef, gather vs scatter summation).
[[nodiscard]] bool u1_cpu_matches_device();

// The device gather force equals the central finite difference of the device
// s_full — validates the per-link re-derivation independently of the CPU.
[[nodiscard]] bool u1_force_matches_fd();

// Leapfrog MD on the link field is time-reversible to roundoff.
[[nodiscard]] bool u1_hmc_reversibility_ok();

// The generic host-free cuda::Hmc instantiates and runs on the gauge action +
// LinkLayout field (momentum sampling / reductions / device accept over links),
// leaving a finite config and a sane acceptance.
[[nodiscard]] bool u1_hmc_runs();

}  // namespace reticolo::cuda
