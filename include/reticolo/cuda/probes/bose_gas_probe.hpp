#pragma once

namespace reticolo::cuda {

// BoseGas (complex scalar) on the device (defined in
// src/cuda/probes/bose_gas_probe.cu). HMC samples the phase-quenched real part
// S_R; the device field element is cplx<T> (AoS), flat-copy-compatible with the
// host Lattice<std::complex<T>>. Validated in both precisions.

// DeviceAction<BoseGas<T>> reproduces the CPU action::BoseGas<T> S_R and F_R on
// a shared config (s_full to roundoff, force bounded) — f64 and f32.
[[nodiscard]] bool bose_gas_cpu_matches_device_f64();
[[nodiscard]] bool bose_gas_cpu_matches_device_f32();

// Leapfrog MD on the complex field (additive drift/kick on the 2-real DOFs) is
// time-reversible to roundoff.
[[nodiscard]] bool bose_gas_hmc_reversibility_ok();

// The generic host-free cuda::Hmc instantiates and runs on BoseGas + cplx field
// (complex momentum sampling, ½Σ|p|² kinetic, device accept) — f64 and f32.
[[nodiscard]] bool bose_gas_hmc_runs_f64();
[[nodiscard]] bool bose_gas_hmc_runs_f32();

}  // namespace reticolo::cuda
