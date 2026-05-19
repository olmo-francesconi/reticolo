#pragma once

#include <cstddef>

namespace reticolo::bench {

// =============================================================================
//  Per-kernel flop-per-dof constants used to convert wall time to a MFLOPS
//  number that's roughly comparable across actions with very different
//  arithmetic densities.
//
//  All constants are *force-evaluation* costs (per dof). `s_full` costs are
//  generally within 10-20% of `compute_force` cost — the same loop shape,
//  just sums a scalar instead of scattering into a force buffer. For the
//  MFLOPS reported here we apply the force-eval flop count to all three
//  kernels (`s_full`, `compute_force`, `compute_force_and_kick`); the rough
//  numbers stay informative for comparing actions, even if individual kernel
//  MFLOPS are off by ~20%.
//
//  k_sin ≈ 15 ops per sin/cos call matches the constant used in
//  `bench_gauge_vs_scalar.cpp`.
// =============================================================================

inline constexpr double k_sin_flops = 15.0;

// ---- scalar actions (per site, per force eval) ------------------------------

inline constexpr double phi4_flops_per_force(int d) noexcept {
    return (2.0 * static_cast<double>(d)) + 7.0;
}
inline constexpr double phi6_flops_per_force(int d) noexcept {
    return (2.0 * static_cast<double>(d)) + 11.0;
}
inline constexpr double sg_flops_per_force(int d) noexcept {
    return (2.0 * static_cast<double>(d)) + 3.0 + k_sin_flops;
}
inline constexpr double xy_flops_per_force(int d) noexcept {
    return 2.0 * static_cast<double>(d) * (1.0 + k_sin_flops);
}
inline constexpr double on_sigma_flops_per_force(int d, int n) noexcept {
    return (2.0 * static_cast<double>(d) * static_cast<double>(n)) +
           static_cast<double>(n * n);
}
inline constexpr double bose_gas_flops_per_force(int d) noexcept {
    // 3d hopping + cosh(μ) anisotropy + per-site polynomial.
    return (3.0 * static_cast<double>(d)) + 12.0 + k_sin_flops;
}

// ---- gauge actions (per link, per force eval) -------------------------------

inline constexpr double u1_flops_per_force_per_link(int d) noexcept {
    double const flops_per_plaq = 3.0 + k_sin_flops + 1.0 + 8.0;
    double const plaqs_per_link = static_cast<double>(d - 1);
    return 0.5 * plaqs_per_link * flops_per_plaq;
}

// 2×2 complex multiply: 8 cmul + 4 cadd = 32 mul + 24 add = 56 flops.
// Per link: 4(d−1) staple cmms + 1 final U·V + TA (~10) + scalar-mul-store.
inline constexpr double su2_flops_per_force_per_link(int d) noexcept {
    constexpr double k_cmm_2x2 = 56.0;
    double const staple_cmms   = 4.0 * static_cast<double>(d - 1);
    return ((staple_cmms + 1.0) * k_cmm_2x2) + 10.0;
}

// 3×3 complex multiply: 27 cmul + 18 cadd = 108 mul + 90 add = 198 flops.
inline constexpr double su3_flops_per_force_per_link(int d) noexcept {
    constexpr double k_cmm_3x3 = 198.0;
    double const staple_cmms   = 4.0 * static_cast<double>(d - 1);
    return ((staple_cmms + 1.0) * k_cmm_3x3) + 18.0;
}

// ---- per-trajectory force-eval count (integrators) --------------------------
// For an integrator running n_md MD steps over τ:
//   Leapfrog:  n_md + 1 force evals
//   Omelyan2:  2·n_md + 1
//   Omelyan4:  4·n_md + 1
inline constexpr double leapfrog_force_evals(int n_md) noexcept {
    return static_cast<double>(n_md) + 1.0;
}
inline constexpr double omelyan2_force_evals(int n_md) noexcept {
    return (2.0 * static_cast<double>(n_md)) + 1.0;
}
inline constexpr double omelyan4_force_evals(int n_md) noexcept {
    return (4.0 * static_cast<double>(n_md)) + 1.0;
}

}  // namespace reticolo::bench
