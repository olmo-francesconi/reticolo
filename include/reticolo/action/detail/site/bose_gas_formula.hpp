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

}  // namespace reticolo::action::detail
