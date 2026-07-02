#pragma once

#include <reticolo/core/cplx.hpp>
#include <reticolo/core/hd.hpp>

// Per-site BoseGas (S_R / F_R) formula — the single source of truth shared by the
// CPU action::BoseGas and the CUDA device_functors<BoseGas>. Only the
// phase-quenched real part is here (what HMC samples); S_I and the LLR combined
// kicks stay CPU-only. The caller supplies the already-weighted neighbour sum
// (the time direction carries cosh(μ)); the loop structure differs CPU↔device
// but the arithmetic does not.
//
//   coef_mass = 2d + m²
//   S_R(x)  = coef_mass·|φ|² + λ|φ|⁴ − 2·Re(conj(φ)·weighted_fwd)
//   F_R(x)  = −2·coef_mass·φ − 4λ|φ|²·φ + 2·staple
//
// weighted_fwd: forward neighbours only (reduce convention), time dir ×cosh(μ).
// staple:       all 2d neighbours, time dir ×cosh(μ).
//
// The imaginary part is μ-independent and touches only the time direction τ (the
// last dim). Same source-of-truth rule: both the CPU action::BoseGas (s_imag /
// compute_force_imag) and the CUDA device kernels call these.
//
//   S_I(x) = 2·Im(conj(φ_x)·φ_{x+τ})           (forward τ neighbour only)
//   F_I(x) = 2i·(φ_{x+τ} − φ_{x−τ})            (both τ neighbours)

namespace reticolo::action::detail {

template <class T>
[[nodiscard]] RETICOLO_HD T
bose_gas_action_site(cplx<T> phi, cplx<T> weighted_fwd, T coef_mass, T lambda) {
    T const abs2 = norm2(phi);
    return (coef_mass * abs2) + (lambda * abs2 * abs2) - (T{2} * re_conj_mul(phi, weighted_fwd));
}

template <class T>
[[nodiscard]] RETICOLO_HD cplx<T>
bose_gas_force_site(cplx<T> phi, cplx<T> staple, T coef_mass, T lambda) {
    T const abs2 = norm2(phi);
    return ((T{-2} * coef_mass) * phi) - ((T{4} * lambda * abs2) * phi) + (T{2} * staple);
}

// Per-site contribution to S_I = 2 Σ_x Im(conj(φ_x)·φ_{x+τ}); the ×2 is folded
// in so a plain sum over sites is S_I directly.
template <class T>
[[nodiscard]] RETICOLO_HD T bose_gas_action_imag_site(cplx<T> phi, cplx<T> phi_fwd_tau) {
    return T{2} * im_conj_mul(phi, phi_fwd_tau);
}

// F_I(x) = 2i·(φ_{x+τ} − φ_{x−τ}). Only the time-direction neighbours contribute.
template <class T>
[[nodiscard]] RETICOLO_HD cplx<T>
bose_gas_force_imag_site(cplx<T> phi_fwd_tau, cplx<T> phi_bwd_tau) {
    return cplx<T>{T{0}, T{2}} * (phi_fwd_tau - phi_bwd_tau);
}

}  // namespace reticolo::action::detail
