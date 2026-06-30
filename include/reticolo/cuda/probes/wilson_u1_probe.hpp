#pragma once

namespace reticolo::cuda {

// Wilson<U(1)> on the device via the SPECIALIZED abelian path (defined in
// src/cuda/probes/wilson_u1_probe.cu). It reuses the CompactU1 angle kernels
// (gauge_u1.cuh) on a 1-angle LinkLayout link — NOT the generic SU(N) matrix
// path. Wilson<U1> and CompactU1 are bit-identical (n_color=1 ⇒ β/N = β), so
// device_functors<Wilson<U1>> is the CompactU1 device path with the action type
// swapped; the host MatrixLinkLattice<U1> flat-copies into the link buffer.

// DeviceAction<Wilson<U1>> reproduces the CPU action::Wilson<U1> s_full and the
// per-link gather force on a shared config (s_full to roundoff, force bounded).
[[nodiscard]] bool wilson_u1_cpu_matches_device();

// Leapfrog MD on the U(1) link field (additive angle drift + kick) is reversible.
[[nodiscard]] bool wilson_u1_hmc_reversibility_ok();

// The generic host-free cuda::Hmc instantiates and runs on Wilson<U1> +
// LinkLayout, leaving a finite config and a sane acceptance.
[[nodiscard]] bool wilson_u1_hmc_runs();

}  // namespace reticolo::cuda
