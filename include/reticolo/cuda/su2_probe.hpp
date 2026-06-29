#pragma once

namespace reticolo::cuda {

// Phase 5 gates (defined in src/cuda/su2_probe.cu): SU(2) Wilson gauge on the
// device. The link field is DeviceField<double, MatrixLayout<SU2>> — the same
// [ndim][nc][nsites] flat order as the host MatrixLinkLattice<SU2>, so a raw
// copy round-trips with no transpose. The generic gauge kernels (gauge_sun.cuh)
// drive the register-local matrix ops through GD = SU2Device; cuda::DeviceAction
// and cuda::Hmc are unchanged — the matrix access pattern lives only in the
// device_functors<Wilson<SU2>> trait and the MatrixLayout drift atom.

// SU2Device's register-local 2×2 ops (mul / mul_adj / adj_mul /
// traceless_antiherm / expi) reproduce math::su2 to roundoff. RETICOLO_HD lets
// the device ops run on the host, so this gate compares them directly.
[[nodiscard]] bool su2_device_ops_match_cpu();

// DeviceAction<Wilson<SU2>> reproduces the CPU action::Wilson<SU2> s_full and
// the staple-gather force on a shared hot config — s_full to roundoff, the
// force to a bounded tolerance (device vs Sleef-batched summation order).
[[nodiscard]] bool su2_cpu_matches_device();

// Leapfrog MD on the matrix field conserves energy: the per-trajectory |ΔH|
// shrinks ≈4× when the step is halved (2nd-order integrator). A scale-free,
// device-only check that the staple force, the additive kick, and the group-exp
// drift form a consistent symplectic pair — catches a wrong drift sign/scale
// that reversibility (symmetric under any reversible drift) would not.
[[nodiscard]] bool su2_energy_conserved_ok();

// Leapfrog MD on the matrix link field (group-exp drift + additive kick) is
// time-reversible to roundoff.
[[nodiscard]] bool su2_hmc_reversibility_ok();

// The device Gell-Mann momentum sampler draws an anti-hermitian algebra element
// per link: the packed reals have ~zero mean and the per-link second moment
// matches the N(0,½)-per-generator kinetic measure the device MH consumes.
[[nodiscard]] bool su2_momentum_moments_ok();

// The generic host-free cuda::Hmc instantiates and runs on Wilson<SU2> +
// MatrixLayout (Gell-Mann sampling / kinetic reduction / group-exp drift /
// device accept over links), leaving a finite config and a sane acceptance.
[[nodiscard]] bool su2_hmc_runs();

}  // namespace reticolo::cuda
