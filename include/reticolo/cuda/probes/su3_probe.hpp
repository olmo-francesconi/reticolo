#pragma once

namespace reticolo::cuda {

// Gates (defined in src/cuda/su3_probe.cu): SU(3) Wilson gauge on the
// device through the SAME generic SU(N) kernels as SU(2) — only the device
// traits change (GD = SU3Device, nc=18, n_gen=8). The link field is
// DeviceField<double, MatrixLayout<SU3>>, the same [ndim][nc][nsites] flat order
// as the host MatrixLinkLattice<SU3>, so a raw copy round-trips with no
// transpose. cuda::DeviceAction and cuda::Hmc are unchanged.

// SU3Device's register-local 3×3 ops (mul / mul_adj / adj_mul /
// traceless_antiherm / expi — Morningstar-Peardon Cayley-Hamilton) reproduce
// math::su3 to roundoff. RETICOLO_HD lets the device ops run on the host.
[[nodiscard]] bool su3_device_ops_match_cpu();

// DeviceAction<Wilson<SU3>> reproduces the CPU action::Wilson<SU3> s_full and
// the staple-gather force on a shared hot config.
[[nodiscard]] bool su3_cpu_matches_device();

// Leapfrog MD on the matrix field conserves energy: |ΔH| shrinks ≈4× when the
// step is halved — a device-only check of the force/kick/drift symplectic pair.
[[nodiscard]] bool su3_energy_conserved_ok();

// Leapfrog MD on the matrix link field is time-reversible to roundoff.
[[nodiscard]] bool su3_hmc_reversibility_ok();

// The device Gell-Mann momentum sampler draws an anti-hermitian algebra element
// per link with the right moments (zero mean, per-link second moment = n_gen).
[[nodiscard]] bool su3_momentum_moments_ok();

// The generic host-free cuda::Hmc instantiates and runs on Wilson<SU3> +
// MatrixLayout, leaving a finite config and a sane acceptance.
[[nodiscard]] bool su3_hmc_runs();

}  // namespace reticolo::cuda
