#pragma once

#include <reticolo/core/sys/hd.hpp>

#include <cmath>
#include <cstddef>

// Single-sourced per-link SU(3) matrix algebra — shared verbatim between the
// CPU path (math::su3, the Sleef-batched slab loops in su3_ops.hpp) and the
// CUDA device path (cuda::SU3Device, cuda/gauge/su3_device.cuh, which forwards
// to these functions). RETICOLO_HD so the exact same body compiles for both
// backends — there is no longer a second hand-copied implementation to drift.
//
// Storage layout (one link element, 18 real doubles):
//
//     For matrix entry (i, j) with i, j ∈ {0, 1, 2}:
//         Re U_{ij}  at slot  2·(3·i + j)
//         Im U_{ij}  at slot  2·(3·i + j) + 1
//
// Matrix exponential follows the Cayley-Hamilton form of Morningstar & Peardon
// (Phys. Rev. D 69, 054501) — for hermitian traceless Q, exp(iQ) =
// f_0·I + f_1·Q + f_2·Q² where f_n are computed from c0 = (1/3)Tr Q³ and
// c1 = (1/2)Tr Q² via the angle θ = acos(c0/c0_max). Small-c1 (Q ≈ 0)
// branch uses Taylor.

namespace reticolo::math::su3 {

// ---------- 18-slot accessors ------------------------------------------------

// Index of the real / imag part of matrix entry (i,j) in the 18-slot layout.
[[gnu::always_inline]] RETICOLO_HD inline constexpr std::size_t idx_re(std::size_t i,
                                                                       std::size_t j) noexcept {
    return 2 * ((3 * i) + j);
}
[[gnu::always_inline]] RETICOLO_HD inline constexpr std::size_t idx_im(std::size_t i,
                                                                       std::size_t j) noexcept {
    return (2 * ((3 * i) + j)) + 1;
}

// ---------- per-site 3×3 complex matrix products ----------------------------
// out, a, b: pointers to 18-real stack arrays. Out must not alias inputs.
// Hand-loops with constant bounds [0,3) — compiler unrolls cleanly.

[[gnu::always_inline]] RETICOLO_HD inline void
mul_3x3(double* out, double const* a, double const* b) noexcept {
    // C_{ij} = sum_k A_{ik} · B_{kj}
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            double cr = 0.0;
            double ci = 0.0;
            for (std::size_t k = 0; k < 3; ++k) {
                double const ar = a[idx_re(i, k)];
                double const ai = a[idx_im(i, k)];
                double const br = b[idx_re(k, j)];
                double const bi = b[idx_im(k, j)];
                cr += (ar * br) - (ai * bi);
                ci += (ar * bi) + (ai * br);
            }
            out[idx_re(i, j)] = cr;
            out[idx_im(i, j)] = ci;
        }
    }
}

// out = a · b†
[[gnu::always_inline]] RETICOLO_HD inline void
mul_adj_3x3(double* out, double const* a, double const* b) noexcept {
    // C_{ij} = sum_k A_{ik} · conj(B_{jk})
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            double cr = 0.0;
            double ci = 0.0;
            for (std::size_t k = 0; k < 3; ++k) {
                double const ar = a[idx_re(i, k)];
                double const ai = a[idx_im(i, k)];
                double const br = b[idx_re(j, k)];
                double const bi = b[idx_im(j, k)];
                // (ar+iai)(br-ibi) = ar·br + ai·bi + i(ai·br - ar·bi)
                cr += (ar * br) + (ai * bi);
                ci += (ai * br) - (ar * bi);
            }
            out[idx_re(i, j)] = cr;
            out[idx_im(i, j)] = ci;
        }
    }
}

// out = a† · b
[[gnu::always_inline]] RETICOLO_HD inline void
adj_mul_3x3(double* out, double const* a, double const* b) noexcept {
    // C_{ij} = sum_k conj(A_{ki}) · B_{kj}
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            double cr = 0.0;
            double ci = 0.0;
            for (std::size_t k = 0; k < 3; ++k) {
                double const ar = a[idx_re(k, i)];
                double const ai = a[idx_im(k, i)];
                double const br = b[idx_re(k, j)];
                double const bi = b[idx_im(k, j)];
                // (ar-iai)(br+ibi) = ar·br + ai·bi + i(ar·bi - ai·br)
                cr += (ar * br) + (ai * bi);
                ci += (ar * bi) - (ai * br);
            }
            out[idx_re(i, j)] = cr;
            out[idx_im(i, j)] = ci;
        }
    }
}

// ---------- traceless anti-hermitian projection (su(3) algebra) -------------
// TA(M) = (M − M†)/2 − (1/3)·Tr((M−M†)/2)·I. Diagonal becomes
// pure imag (Im_{ii} − T/3); off-diag is the anti-hermitian completion.
[[gnu::always_inline]] RETICOLO_HD inline void traceless_antiherm_3x3(double* out,
                                                                      double const* in) noexcept {
    double const im00     = in[idx_im(0, 0)];
    double const im11     = in[idx_im(1, 1)];
    double const im22     = in[idx_im(2, 2)];
    double const t_over_3 = (im00 + im11 + im22) / 3.0;
    out[idx_re(0, 0)]     = 0.0;
    out[idx_im(0, 0)]     = im00 - t_over_3;
    out[idx_re(1, 1)]     = 0.0;
    out[idx_im(1, 1)]     = im11 - t_over_3;
    out[idx_re(2, 2)]     = 0.0;
    out[idx_im(2, 2)]     = im22 - t_over_3;
    // (M − M†)/2 off-diagonal: ((Re ij - Re ji)/2, (Im ij + Im ji)/2).
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = i + 1; j < 3; ++j) {
            double const r_ij = in[idx_re(i, j)];
            double const i_ij = in[idx_im(i, j)];
            double const r_ji = in[idx_re(j, i)];
            double const i_ji = in[idx_im(j, i)];
            double const re   = 0.5 * (r_ij - r_ji);
            double const im   = 0.5 * (i_ij + i_ji);
            out[idx_re(i, j)] = re;
            out[idx_im(i, j)] = im;
            out[idx_re(j, i)] = -re;
            out[idx_im(j, i)] = im;
        }
    }
}

// ---------- Cayley-Hamilton matrix exponential (Morningstar-Peardon) --------
//
// Build V = exp(dt · P) where P is anti-hermitian traceless 3×3 stored in the
// 18-real layout. Internally form Q = -i·dt·P (hermitian, traceless) and use
// Morningstar-Peardon's coefficients:
//
//  exp(iQ) = f_0·I + f_1·Q + f_2·Q²        (Cayley-Hamilton)
//
// where (with c1 = (1/2)Tr Q², c0 = (1/3)Tr Q³, c0_max = 2·(c1/3)^{3/2},
//       θ = acos(c0/c0_max),
//       u = sqrt(c1/3)·cos(θ/3), w = sqrt(c1)·sin(θ/3),
//       ξ(w) = sin(w)/w,  den = 9u² − w²):
//
//  h_0 = (u² − w²)·e^{2iu} + e^{−iu}·[8 u² cos(w) + 2 i u (3u² + w²) ξ(w)]
//  h_1 = 2u·e^{2iu} − e^{−iu}·[2 u cos(w) − i (3u² − w²) ξ(w)]
//  h_2 = e^{2iu} − e^{−iu}·[cos(w) + 3 i u ξ(w)]
//  f_n = h_n / den
//
// Small-c1 (Q ≈ 0) branch: Taylor series exp(iQ) ≈ I + iQ − Q²/2 − iQ³/6.
// For HMC step sizes this branch is hit only when ‖dt·P‖ is below ~1e-3.
[[gnu::always_inline]] RETICOLO_HD inline void
exp_su3(double* v, double const* p, double dt) noexcept {
    // Q = -i · dt · P, with P anti-hermitian (P_re_diag=0, structural).
    //   Q_re = dt · P_im,  Q_im = -dt · P_re.
    double q[18];
    for (std::size_t k = 0; k < 9; ++k) {
        q[2 * k]       = dt * p[(2 * k) + 1];
        q[(2 * k) + 1] = -dt * p[2 * k];
    }

    // c1 = (1/2) Tr Q² = (1/2)·sum_{ij} Q_{ij}·Q_{ji}. For hermitian Q,
    // Q_{ji} = conj(Q_{ij}), so Q_{ij}·Q_{ji} = |Q_{ij}|². So c1 reduces to
    // (1/2)·sum_{ij} |Q_{ij}|² = (1/2)·‖Q‖²_F (Frobenius squared / 2).
    double c1 = 0.0;
    for (double const k : q) {
        c1 += k * k;
    }
    c1 *= 0.5;

    constexpr double k_small_c1 = 1.0e-8;
    if (c1 < k_small_c1) {
        // Taylor branch: V = I + iQ - Q²/2 - iQ³/6 + Q⁴/24.
        double q2[18];
        mul_3x3(q2, q, q);
        double q3[18];
        mul_3x3(q3, q2, q);
        for (std::size_t k = 0; k < 18; ++k) {
            v[k] = -0.5 * q2[k];
        }
        v[idx_re(0, 0)] += 1.0;
        v[idx_re(1, 1)] += 1.0;
        v[idx_re(2, 2)] += 1.0;
        // Add i·Q (Re += -Im Q, Im += +Re Q).
        for (std::size_t k = 0; k < 9; ++k) {
            v[2 * k] += -q[(2 * k) + 1];
            v[(2 * k) + 1] += q[2 * k];
        }
        // Add -i·Q³/6 (Re += Im Q³/6, Im += -Re Q³/6).
        constexpr double k_inv6 = 1.0 / 6.0;
        for (std::size_t k = 0; k < 9; ++k) {
            v[2 * k] += k_inv6 * q3[(2 * k) + 1];
            v[(2 * k) + 1] -= k_inv6 * q3[2 * k];
        }
        return;
    }

    // c0 = (1/3) Tr Q³ = det Q for hermitian traceless 3×3.
    // For hermitian Q, c0 is real. Compute via Q² · Q diagonal trace.
    double q2[18];
    mul_3x3(q2, q, q);
    double const tr_q3 = (q2[idx_re(0, 0)] * q[idx_re(0, 0)] - q2[idx_im(0, 0)] * q[idx_im(0, 0)]) +
                         (q2[idx_re(0, 1)] * q[idx_re(1, 0)] - q2[idx_im(0, 1)] * q[idx_im(1, 0)]) +
                         (q2[idx_re(0, 2)] * q[idx_re(2, 0)] - q2[idx_im(0, 2)] * q[idx_im(2, 0)]) +
                         (q2[idx_re(1, 0)] * q[idx_re(0, 1)] - q2[idx_im(1, 0)] * q[idx_im(0, 1)]) +
                         (q2[idx_re(1, 1)] * q[idx_re(1, 1)] - q2[idx_im(1, 1)] * q[idx_im(1, 1)]) +
                         (q2[idx_re(1, 2)] * q[idx_re(2, 1)] - q2[idx_im(1, 2)] * q[idx_im(2, 1)]) +
                         (q2[idx_re(2, 0)] * q[idx_re(0, 2)] - q2[idx_im(2, 0)] * q[idx_im(0, 2)]) +
                         (q2[idx_re(2, 1)] * q[idx_re(1, 2)] - q2[idx_im(2, 1)] * q[idx_im(1, 2)]) +
                         (q2[idx_re(2, 2)] * q[idx_re(2, 2)] - q2[idx_im(2, 2)] * q[idx_im(2, 2)]);
    double const c0    = tr_q3 / 3.0;

    double const c1_over_3 = c1 / 3.0;
    double const c0_max    = 2.0 * c1_over_3 * std::sqrt(c1_over_3);
    // c0/c0_max should lie in [-1, 1]. Clamp against tiny numerical overshoot.
    // Written as explicit min-then-max ternaries (not std::max/std::min) so the
    // exact same body compiles under nvcc without --expt-relaxed-constexpr;
    // this also fixes the NaN behaviour to match std::min/std::max (NaN input
    // clamps to 1.0) rather than a naive ternary chain (which would pass NaN
    // through) — the one intentional semantic change from the old hand-copied
    // device version.
    double ratio = 0.0;
    if (c0_max > 0.0) {
        double r = c0 / c0_max;
        r        = (r < 1.0) ? r : 1.0;    // std::min(1.0, r)  → NaN yields 1.0
        r        = (-1.0 < r) ? r : -1.0;  // std::max(-1.0, r)
        ratio    = r;
    }
    double const theta = std::acos(ratio);

    double const sqrt_c1_3 = std::sqrt(c1_over_3);
    double const u         = sqrt_c1_3 * std::cos(theta / 3.0);
    double const w         = std::sqrt(c1) * std::sin(theta / 3.0);

    double const cw  = std::cos(w);
    double const cu  = std::cos(u);
    double const su_ = std::sin(u);
    double const c2u = std::cos(2.0 * u);
    double const s2u = std::sin(2.0 * u);
    double const xi  = (std::abs(w) > 1.0e-10) ? (std::sin(w) / w) : 1.0;

    double const u2  = u * u;
    double const w2  = w * w;
    double const den = (9.0 * u2) - w2;

    // h_n (complex): h_n = h_n_re + i h_n_im.
    // h_0 = (u²−w²) e^{2iu} + e^{-iu} · [8 u² cw + 2 i u (3u² + w²) ξ]
    double const t0a_re = (u2 - w2) * c2u;
    double const t0a_im = (u2 - w2) * s2u;
    double const t0b_re = (cu * (8.0 * u2 * cw)) + (su_ * (2.0 * u * (3.0 * u2 + w2) * xi));
    double const t0b_im = -(su_ * (8.0 * u2 * cw)) + (cu * (2.0 * u * (3.0 * u2 + w2) * xi));
    double const h0_re  = t0a_re + t0b_re;
    double const h0_im  = t0a_im + t0b_im;

    // h_1 = 2u e^{2iu} − e^{-iu} · [2 u cw − i (3u² − w²) ξ]
    double const t1a_re    = 2.0 * u * c2u;
    double const t1a_im    = 2.0 * u * s2u;
    double const inner1_re = 2.0 * u * cw;
    double const inner1_im = -((3.0 * u2) - w2) * xi;
    // e^{-iu} · inner1 = (cu - i·su) · (inner_re + i·inner_im)
    //                  = (cu·inner_re + su·inner_im) + i·(cu·inner_im - su·inner_re)
    double const t1b_re = (cu * inner1_re) + (su_ * inner1_im);
    double const t1b_im = (cu * inner1_im) - (su_ * inner1_re);
    double const h1_re  = t1a_re - t1b_re;
    double const h1_im  = t1a_im - t1b_im;

    // h_2 = e^{2iu} − e^{-iu} · [cw + 3 i u ξ]
    double const inner2_re = cw;
    double const inner2_im = 3.0 * u * xi;
    double const t2b_re    = (cu * inner2_re) + (su_ * inner2_im);
    double const t2b_im    = (cu * inner2_im) - (su_ * inner2_re);
    double const h2_re     = c2u - t2b_re;
    double const h2_im     = s2u - t2b_im;

    double const inv_den = 1.0 / den;
    double const f0_re   = h0_re * inv_den;
    double const f0_im   = h0_im * inv_den;
    double const f1_re   = h1_re * inv_den;
    double const f1_im   = h1_im * inv_den;
    double const f2_re   = h2_re * inv_den;
    double const f2_im   = h2_im * inv_den;

    // exp(iQ) = f_0·I + f_1·Q + f_2·Q². Build V entry-by-entry.
    for (std::size_t k = 0; k < 9; ++k) {
        std::size_t const kr = 2 * k;
        std::size_t const ki = kr + 1;
        double const q_re    = q[kr];
        double const q_im    = q[ki];
        double const q2_re   = q2[kr];
        double const q2_im   = q2[ki];
        v[kr] = ((f1_re * q_re) - (f1_im * q_im)) + ((f2_re * q2_re) - (f2_im * q2_im));
        v[ki] = ((f1_re * q_im) + (f1_im * q_re)) + ((f2_re * q2_im) + (f2_im * q2_re));
    }
    // Add f_0·I to the diagonal.
    v[idx_re(0, 0)] += f0_re;
    v[idx_im(0, 0)] += f0_im;
    v[idx_re(1, 1)] += f0_re;
    v[idx_im(1, 1)] += f0_im;
    v[idx_re(2, 2)] += f0_re;
    v[idx_im(2, 2)] += f0_im;
}

}  // namespace reticolo::math::su3
