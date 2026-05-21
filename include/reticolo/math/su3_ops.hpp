#pragma once

#include <reticolo/math/vec_libm.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace reticolo::math::su3 {

// Hand-written 3×3 complex matrix kernels for SU(3) lattice gauge fields.
//
// Storage layout (one link element, 18 real doubles):
//
//     For matrix entry (i, j) with i, j ∈ {0, 1, 2}:
//         Re U_{ij}  at slot  2·(3·i + j)
//         Im U_{ij}  at slot  2·(3·i + j) + 1
//
//     So the 18 slots in order are:
//         (00r 00i 01r 01i 02r 02i 10r 10i 11r 11i 12r 12i 20r 20i 21r 21i 22r 22i)
//
// Slab kernels read/write nc=18 stride-1 component arrays per direction; the
// outer per-site loop auto-vectorises (16+ stride-1 streams, hand-unrolled
// inner products via the small 3×3-with-constant-bounds loops below).
//
// Algebra (anti-hermitian P, su(3)) is stored in the same 18-real layout
// with the structural constraints: Re P_{ii} = 0 for i=0,1,2; P_{ji} = -conj(P_{ij})
// for i ≠ j; sum_i Im P_{ii} = 0 (traceless). Eight independent real
// parameters per link (Gell-Mann coordinates h_1..h_8). See sample_algebra_slab
// for the explicit parameterisation P = i·Σ_a h_a·λ_a (Gell-Mann basis with
// Tr(λ_a λ_b) = 2 δ_ab — no extra 1/2 in the basis), giving K_per_link =
// ‖h‖² and Q(P) ∝ exp(−K) when h_a ~ N(0, 1/√2) — same convention as SU(2).
//
// Matrix exponential follows the Cayley-Hamilton form of Morningstar & Peardon
// (Phys. Rev. D 69, 054501) — for hermitian traceless Q, exp(iQ) =
// f_0·I + f_1·Q + f_2·Q² where f_n are computed from c0 = (1/3)Tr Q³ and
// c1 = (1/2)Tr Q² via the angle θ = acos(c0/c0_max). Small-c1 (Q ≈ 0)
// branch uses Taylor.

// ---------- 18-slot accessors ------------------------------------------------

// Index of the real / imag part of matrix entry (i,j) in the 18-slot layout.
[[gnu::always_inline]] inline constexpr std::size_t idx_re(std::size_t i, std::size_t j) noexcept {
    return 2 * ((3 * i) + j);
}
[[gnu::always_inline]] inline constexpr std::size_t idx_im(std::size_t i, std::size_t j) noexcept {
    return (2 * ((3 * i) + j)) + 1;
}

// ---------- per-site 3×3 complex matrix products ----------------------------
// out, a, b: pointers to 18-real stack arrays. Out must not alias inputs.
// Hand-loops with constant bounds [0,3) — compiler unrolls cleanly.

[[gnu::always_inline]] inline void mul_3x3(double* out, double const* a, double const* b) noexcept {
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
[[gnu::always_inline]] inline void
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
[[gnu::always_inline]] inline void
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
[[gnu::always_inline]] inline void traceless_antiherm_3x3(double* out, double const* in) noexcept {
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
[[gnu::always_inline]] inline void exp_su3(double* v, double const* p, double dt) noexcept {
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
    for (std::size_t k = 0; k < 18; ++k) {
        c1 += q[k] * q[k];
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
    double const c0 = tr_q3 / 3.0;

    double const c1_over_3 = c1 / 3.0;
    double const c0_max    = 2.0 * c1_over_3 * std::sqrt(c1_over_3);
    // c0/c0_max should lie in [-1, 1]. Clamp against tiny numerical overshoot.
    double const ratio = (c0_max > 0.0) ? std::max(-1.0, std::min(1.0, c0 / c0_max)) : 0.0;
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

// ---------- project_su3: reunitarise via Gram-Schmidt + cross product -------
[[gnu::always_inline]] inline void project_su3(double* m) noexcept {
    // Row 0: load, normalise.
    double r0[6];
    for (std::size_t j = 0; j < 3; ++j) {
        r0[2 * j]       = m[idx_re(0, j)];
        r0[(2 * j) + 1] = m[idx_im(0, j)];
    }
    double n0_sq = 0.0;
    for (std::size_t k = 0; k < 6; ++k) {
        n0_sq += r0[k] * r0[k];
    }
    double const inv_n0 = 1.0 / std::sqrt(n0_sq);
    for (std::size_t k = 0; k < 6; ++k) {
        r0[k] *= inv_n0;
    }

    // Row 1: subtract <r0, r1>·r0 then normalise.
    double r1[6];
    for (std::size_t j = 0; j < 3; ++j) {
        r1[2 * j]       = m[idx_re(1, j)];
        r1[(2 * j) + 1] = m[idx_im(1, j)];
    }
    // <r0, r1> = sum_j conj(r0_j) · r1_j
    double dot_re = 0.0;
    double dot_im = 0.0;
    for (std::size_t j = 0; j < 3; ++j) {
        double const ar = r0[2 * j];
        double const ai = r0[(2 * j) + 1];
        double const br = r1[2 * j];
        double const bi = r1[(2 * j) + 1];
        // conj(a)·b = (ar - i·ai)(br + i·bi) = (ar·br + ai·bi) + i·(ar·bi − ai·br)
        dot_re += (ar * br) + (ai * bi);
        dot_im += (ar * bi) - (ai * br);
    }
    for (std::size_t j = 0; j < 3; ++j) {
        double const ar = r0[2 * j];
        double const ai = r0[(2 * j) + 1];
        // r1 -= dot · r0  (dot is complex, multiply)
        r1[2 * j] -= (dot_re * ar) - (dot_im * ai);
        r1[(2 * j) + 1] -= (dot_re * ai) + (dot_im * ar);
    }
    double n1_sq = 0.0;
    for (std::size_t k = 0; k < 6; ++k) {
        n1_sq += r1[k] * r1[k];
    }
    double const inv_n1 = 1.0 / std::sqrt(n1_sq);
    for (std::size_t k = 0; k < 6; ++k) {
        r1[k] *= inv_n1;
    }

    // Row 2 = conj(r0 × r1) so that det = +1.
    // (a × b)_k = ε_{kij}·a_i·b_j. For k=0: a_1·b_2 - a_2·b_1, etc.
    auto cmul_sub = [](double ar,
                       double ai,
                       double br,
                       double bi,
                       double cr,
                       double ci,
                       double dr,
                       double di,
                       double& out_re,
                       double& out_im) noexcept {
        out_re = (ar * br) - (ai * bi) - ((cr * dr) - (ci * di));
        out_im = (ar * bi) + (ai * br) - ((cr * di) + (ci * dr));
    };
    double r2[6];
    cmul_sub(r0[2], r0[3], r1[4], r1[5], r0[4], r0[5], r1[2], r1[3], r2[0], r2[1]);  // (k=0)
    cmul_sub(r0[4], r0[5], r1[0], r1[1], r0[0], r0[1], r1[4], r1[5], r2[2], r2[3]);  // (k=1)
    cmul_sub(r0[0], r0[1], r1[2], r1[3], r0[2], r0[3], r1[0], r1[1], r2[4], r2[5]);  // (k=2)
    // Conjugate each entry.
    for (std::size_t j = 0; j < 3; ++j) {
        r2[(2 * j) + 1] = -r2[(2 * j) + 1];
    }

    // Write rows back.
    for (std::size_t j = 0; j < 3; ++j) {
        m[idx_re(0, j)] = r0[2 * j];
        m[idx_im(0, j)] = r0[(2 * j) + 1];
        m[idx_re(1, j)] = r1[2 * j];
        m[idx_im(1, j)] = r1[(2 * j) + 1];
        m[idx_re(2, j)] = r2[2 * j];
        m[idx_im(2, j)] = r2[(2 * j) + 1];
    }
}

// ---------- slab drivers ----------------------------------------------------

[[gnu::always_inline]] inline void
mul_slab(double* out, double const* a, double const* b, std::size_t n) noexcept {
    for (std::size_t s = 0; s < n; ++s) {
        double a_s[18];
        double b_s[18];
        for (std::size_t k = 0; k < 18; ++k) {
            a_s[k] = a[(k * n) + s];
            b_s[k] = b[(k * n) + s];
        }
        double o_s[18];
        mul_3x3(o_s, a_s, b_s);
        for (std::size_t k = 0; k < 18; ++k) {
            out[(k * n) + s] = o_s[k];
        }
    }
}

[[gnu::always_inline]] inline void
mul_adj_slab(double* out, double const* a, double const* b, std::size_t n) noexcept {
    for (std::size_t s = 0; s < n; ++s) {
        double a_s[18];
        double b_s[18];
        for (std::size_t k = 0; k < 18; ++k) {
            a_s[k] = a[(k * n) + s];
            b_s[k] = b[(k * n) + s];
        }
        double o_s[18];
        mul_adj_3x3(o_s, a_s, b_s);
        for (std::size_t k = 0; k < 18; ++k) {
            out[(k * n) + s] = o_s[k];
        }
    }
}

[[gnu::always_inline]] inline void
adj_mul_slab(double* out, double const* a, double const* b, std::size_t n) noexcept {
    for (std::size_t s = 0; s < n; ++s) {
        double a_s[18];
        double b_s[18];
        for (std::size_t k = 0; k < 18; ++k) {
            a_s[k] = a[(k * n) + s];
            b_s[k] = b[(k * n) + s];
        }
        double o_s[18];
        adj_mul_3x3(o_s, a_s, b_s);
        for (std::size_t k = 0; k < 18; ++k) {
            out[(k * n) + s] = o_s[k];
        }
    }
}

// In-place U ← exp(dt·P) · U, slab edition.
//
// The per-site Cayley-Hamilton exp has 1 acos + 5 sincos calls per link plus
// 2 sqrt. Batching them across all slab sites via Sleef cuts the
// transcendental cost ~2× on AArch64 / AVX2. The structure is multi-pass:
//
//  pass 1 (per site, scalar): compute c1, c0, branch flag, ratio = c0/c0_max
//                             (clamped + safe for small-c1), √(c1/3), √c1.
//  pass 2 (vector):           theta = acos(ratio) via acos_batch.
//  pass 3 (vector):           build theta/3 buffer; sincos_batch → (s_t3, c_t3).
//  pass 3.5 (per site):       u = √(c1/3)·c_t3, w = √c1·s_t3.
//  pass 4 (vector):           sincos_batch(u) → (su, cu), sincos_batch(w) → (sw, cw).
//  pass 5 (per site):         assemble V = f₀·I + f₁·Q + f₂·Q² (main branch)
//                             or fall back to Taylor for sites where small_flag = 1;
//                             then U ← V · U.
//
// 2u-trig is computed inline as c2u = cu² − su² and s2u = 2·cu·su (no extra
// transcendentals). ξ = sin(w)/w uses a branchless small-w guard.
[[gnu::always_inline]] inline void
expi_lmul_slab(double* u, double const* p, double dt, std::size_t n) noexcept {
    thread_local std::vector<double> scratch;
    constexpr std::size_t k_slabs = 13;
    if (scratch.size() < k_slabs * n) {
        scratch.resize(k_slabs * n);
    }
    double* const ratio_buf  = scratch.data() + (0 * n);  // pass1→pass2
    double* const theta_buf  = scratch.data() + (1 * n);  // pass2→pass3 prep
    double* const t3_buf     = scratch.data() + (2 * n);  // pass3 input
    double* const sqrt_c1_3  = scratch.data() + (3 * n);
    double* const sqrt_c1    = scratch.data() + (4 * n);
    double* const small_flag = scratch.data() + (5 * n);
    double* const u_buf      = scratch.data() + (6 * n);
    double* const w_buf      = scratch.data() + (7 * n);
    double* const cu_buf     = scratch.data() + (8 * n);
    double* const su_buf     = scratch.data() + (9 * n);
    double* const cw_buf     = scratch.data() + (10 * n);
    double* const sw_buf     = scratch.data() + (11 * n);
    double* const s_t3_buf   = scratch.data() + (12 * n);

    constexpr double k_small_c1 = 1.0e-8;
    constexpr double k_eps_w    = 1.0e-10;
    constexpr double k_inv_3    = 1.0 / 3.0;

    // ----- Pass 1: per-site scalar prep ----------------------------------
    // Read p_s[18], build q = -i·dt·p, compute c1, c0, c0_max, clamped ratio
    // and the two √c1 forms. Mark small-c1 sites for the Taylor fallback.
    for (std::size_t s = 0; s < n; ++s) {
        double q[18];
        for (std::size_t k = 0; k < 9; ++k) {
            q[2 * k]       = dt * p[((2 * k) + 1) * n + s];
            q[(2 * k) + 1] = -dt * p[(2 * k) * n + s];
        }
        double c1 = 0.0;
        for (std::size_t k = 0; k < 18; ++k) {
            c1 += q[k] * q[k];
        }
        c1 *= 0.5;
        if (c1 < k_small_c1) {
            small_flag[s] = 1.0;
            ratio_buf[s]  = 0.0;
            sqrt_c1_3[s]  = 0.0;
            sqrt_c1[s]    = 0.0;
            continue;
        }
        small_flag[s] = 0.0;
        double q2[18];
        mul_3x3(q2, q, q);
        // c0 = Tr(Q³)/3 = Tr(Q² · Q)/3, real for hermitian Q.
        double const tr_q3 =
            (q2[idx_re(0, 0)] * q[idx_re(0, 0)] - q2[idx_im(0, 0)] * q[idx_im(0, 0)]) +
            (q2[idx_re(0, 1)] * q[idx_re(1, 0)] - q2[idx_im(0, 1)] * q[idx_im(1, 0)]) +
            (q2[idx_re(0, 2)] * q[idx_re(2, 0)] - q2[idx_im(0, 2)] * q[idx_im(2, 0)]) +
            (q2[idx_re(1, 0)] * q[idx_re(0, 1)] - q2[idx_im(1, 0)] * q[idx_im(0, 1)]) +
            (q2[idx_re(1, 1)] * q[idx_re(1, 1)] - q2[idx_im(1, 1)] * q[idx_im(1, 1)]) +
            (q2[idx_re(1, 2)] * q[idx_re(2, 1)] - q2[idx_im(1, 2)] * q[idx_im(2, 1)]) +
            (q2[idx_re(2, 0)] * q[idx_re(0, 2)] - q2[idx_im(2, 0)] * q[idx_im(0, 2)]) +
            (q2[idx_re(2, 1)] * q[idx_re(1, 2)] - q2[idx_im(2, 1)] * q[idx_im(1, 2)]) +
            (q2[idx_re(2, 2)] * q[idx_re(2, 2)] - q2[idx_im(2, 2)] * q[idx_im(2, 2)]);
        double const c0        = tr_q3 * k_inv_3;
        double const c1_over_3 = c1 * k_inv_3;
        double const sc13      = std::sqrt(c1_over_3);
        double const sc1       = std::sqrt(c1);
        double const c0_max    = 2.0 * c1_over_3 * sc13;
        double const ratio     = (c0_max > 0.0) ? std::max(-1.0, std::min(1.0, c0 / c0_max)) : 0.0;
        ratio_buf[s]           = ratio;
        sqrt_c1_3[s]           = sc13;
        sqrt_c1[s]             = sc1;
    }

    // ----- Pass 2: theta = acos(ratio) -----------------------------------
    reticolo::math::acos_batch(theta_buf, ratio_buf, n);

    // ----- Pass 3: theta/3 → sincos --------------------------------------
    for (std::size_t s = 0; s < n; ++s) {
        t3_buf[s] = theta_buf[s] * k_inv_3;
    }
    reticolo::math::sincos_batch(s_t3_buf, /*cos*/ theta_buf, t3_buf, n);
    //   ↑ reuse theta_buf for cos(theta/3) — its acos contents are no longer
    //     needed. Calling convention of sincos_batch is (sin_dst, cos_dst, src, n).

    // ----- Pass 3.5: u, w ------------------------------------------------
    for (std::size_t s = 0; s < n; ++s) {
        double const c_t3 = theta_buf[s];
        double const s_t3 = s_t3_buf[s];
        u_buf[s]          = sqrt_c1_3[s] * c_t3;
        w_buf[s]          = sqrt_c1[s] * s_t3;
    }

    // ----- Pass 4: sincos(u), sincos(w) ----------------------------------
    reticolo::math::sincos_batch(su_buf, cu_buf, u_buf, n);
    reticolo::math::sincos_batch(sw_buf, cw_buf, w_buf, n);

    // ----- Pass 5: per-site assembly + multiply --------------------------
    for (std::size_t s = 0; s < n; ++s) {
        // Recompute q (cheap vs stashing 18·n doubles).
        double q[18];
        for (std::size_t k = 0; k < 9; ++k) {
            q[2 * k]       = dt * p[((2 * k) + 1) * n + s];
            q[(2 * k) + 1] = -dt * p[(2 * k) * n + s];
        }
        double u_old[18];
        for (std::size_t k = 0; k < 18; ++k) {
            u_old[k] = u[(k * n) + s];
        }

        double v[18];
        if (small_flag[s] != 0.0) {
            // Taylor branch: V = I + iQ − Q²/2 − iQ³/6.
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
            for (std::size_t k = 0; k < 9; ++k) {
                v[2 * k] += -q[(2 * k) + 1];
                v[(2 * k) + 1] += q[2 * k];
            }
            constexpr double k_inv6 = 1.0 / 6.0;
            for (std::size_t k = 0; k < 9; ++k) {
                v[2 * k] += k_inv6 * q3[(2 * k) + 1];
                v[(2 * k) + 1] -= k_inv6 * q3[2 * k];
            }
        } else {
            double const u_ = u_buf[s];
            double const w_ = w_buf[s];
            double const cu = cu_buf[s];
            double const su = su_buf[s];
            double const cw = cw_buf[s];
            double const sw = sw_buf[s];
            // 2u trig via doubling — no extra Sleef call.
            double const c2u = (cu * cu) - (su * su);
            double const s2u = 2.0 * cu * su;
            // ξ(w) = sin(w)/w, branchless small-w guard.
            double const xi = (std::abs(w_) > k_eps_w) ? (sw / w_) : 1.0;

            double const u2  = u_ * u_;
            double const w2  = w_ * w_;
            double const den = (9.0 * u2) - w2;

            // h_0 = (u²−w²) e^{2iu} + e^{-iu}·[8 u² cw + 2 i u (3u² + w²) ξ]
            double const t0a_re = (u2 - w2) * c2u;
            double const t0a_im = (u2 - w2) * s2u;
            double const inner0 = 2.0 * u_ * (3.0 * u2 + w2) * xi;
            double const t0b_re = (cu * (8.0 * u2 * cw)) + (su * inner0);
            double const t0b_im = -(su * (8.0 * u2 * cw)) + (cu * inner0);
            double const h0_re  = t0a_re + t0b_re;
            double const h0_im  = t0a_im + t0b_im;

            // h_1 = 2u e^{2iu} − e^{-iu}·[2 u cw − i (3u² − w²) ξ]
            double const t1a_re    = 2.0 * u_ * c2u;
            double const t1a_im    = 2.0 * u_ * s2u;
            double const inner1_re = 2.0 * u_ * cw;
            double const inner1_im = -((3.0 * u2) - w2) * xi;
            double const t1b_re    = (cu * inner1_re) + (su * inner1_im);
            double const t1b_im    = (cu * inner1_im) - (su * inner1_re);
            double const h1_re     = t1a_re - t1b_re;
            double const h1_im     = t1a_im - t1b_im;

            // h_2 = e^{2iu} − e^{-iu}·[cw + 3 i u ξ]
            double const inner2_re = cw;
            double const inner2_im = 3.0 * u_ * xi;
            double const t2b_re    = (cu * inner2_re) + (su * inner2_im);
            double const t2b_im    = (cu * inner2_im) - (su * inner2_re);
            double const h2_re     = c2u - t2b_re;
            double const h2_im     = s2u - t2b_im;

            double const inv_den = 1.0 / den;
            double const f0_re   = h0_re * inv_den;
            double const f0_im   = h0_im * inv_den;
            double const f1_re   = h1_re * inv_den;
            double const f1_im   = h1_im * inv_den;
            double const f2_re   = h2_re * inv_den;
            double const f2_im   = h2_im * inv_den;

            double q2[18];
            mul_3x3(q2, q, q);
            // V = f0·I + f1·Q + f2·Q²
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
            v[idx_re(0, 0)] += f0_re;
            v[idx_im(0, 0)] += f0_im;
            v[idx_re(1, 1)] += f0_re;
            v[idx_im(1, 1)] += f0_im;
            v[idx_re(2, 2)] += f0_re;
            v[idx_im(2, 2)] += f0_im;
        }

        double o_s[18];
        mul_3x3(o_s, v, u_old);
        for (std::size_t k = 0; k < 18; ++k) {
            u[(k * n) + s] = o_s[k];
        }
    }
}

[[gnu::always_inline]] inline void project_slab(double* u, std::size_t n) noexcept {
    for (std::size_t s = 0; s < n; ++s) {
        double m[18];
        for (std::size_t k = 0; k < 18; ++k) {
            m[k] = u[(k * n) + s];
        }
        project_su3(m);
        for (std::size_t k = 0; k < 18; ++k) {
            u[(k * n) + s] = m[k];
        }
    }
}

// Sample P from the anti-hermitian-traceless Gaussian on su(3). 8 Gell-Mann
// coords h_1..h_8 ~ N(0, 1/√2) packed into the 18-real layout via
// P = i·sum_a h_a·λ_a (Tr(λ_a λ_b) = 2 δ_ab convention, no extra 1/2 in
// the basis). Gives Q(P) ∝ exp(−K) with K = (1/2) Tr(P† P) = ‖h‖² — same
// matching convention as SU(2).
template <class Rng>
[[gnu::always_inline]] inline void
sample_algebra_slab(double* p, Rng& rng, std::size_t n) noexcept {
    constexpr double k_inv_sqrt2 = 0.70710678118654752440;
    constexpr double k_inv_sqrt3 = 0.57735026918962576451;
    // Pre-fill 8n independent N(0, 1/√2) draws into a thread-local buffer
    // then scatter — same shape as SU(2)::sample_algebra_slab.
    thread_local std::vector<double> h_buf;
    if (h_buf.size() < 8 * n) {
        h_buf.resize(8 * n);
    }
    rng.normal_fill(h_buf.data(), 8 * n);
    double const* const h1_arr = h_buf.data() + (0 * n);
    double const* const h2_arr = h_buf.data() + (1 * n);
    double const* const h3_arr = h_buf.data() + (2 * n);
    double const* const h4_arr = h_buf.data() + (3 * n);
    double const* const h5_arr = h_buf.data() + (4 * n);
    double const* const h6_arr = h_buf.data() + (5 * n);
    double const* const h7_arr = h_buf.data() + (6 * n);
    double const* const h8_arr = h_buf.data() + (7 * n);
    for (std::size_t s = 0; s < n; ++s) {
        double const h1            = h1_arr[s] * k_inv_sqrt2;
        double const h2            = h2_arr[s] * k_inv_sqrt2;
        double const h3            = h3_arr[s] * k_inv_sqrt2;
        double const h4            = h4_arr[s] * k_inv_sqrt2;
        double const h5            = h5_arr[s] * k_inv_sqrt2;
        double const h6            = h6_arr[s] * k_inv_sqrt2;
        double const h7            = h7_arr[s] * k_inv_sqrt2;
        double const h8            = h8_arr[s] * k_inv_sqrt2;
        double const h8_over_sqrt3 = h8 * k_inv_sqrt3;
        // Diagonal: 00 = i·(h3 + h8/√3), 11 = i·(-h3 + h8/√3), 22 = i·(-2 h8/√3)
        p[(idx_re(0, 0) * n) + s] = 0.0;
        p[(idx_im(0, 0) * n) + s] = h3 + h8_over_sqrt3;
        p[(idx_re(1, 1) * n) + s] = 0.0;
        p[(idx_im(1, 1) * n) + s] = -h3 + h8_over_sqrt3;
        p[(idx_re(2, 2) * n) + s] = 0.0;
        p[(idx_im(2, 2) * n) + s] = -2.0 * h8_over_sqrt3;
        // Off-diagonal: P_{ij} = h_re + i·h_im, P_{ji} = -h_re + i·h_im.
        p[(idx_re(0, 1) * n) + s] = h2;
        p[(idx_im(0, 1) * n) + s] = h1;
        p[(idx_re(1, 0) * n) + s] = -h2;
        p[(idx_im(1, 0) * n) + s] = h1;
        p[(idx_re(0, 2) * n) + s] = h5;
        p[(idx_im(0, 2) * n) + s] = h4;
        p[(idx_re(2, 0) * n) + s] = -h5;
        p[(idx_im(2, 0) * n) + s] = h4;
        p[(idx_re(1, 2) * n) + s] = h7;
        p[(idx_im(1, 2) * n) + s] = h6;
        p[(idx_re(2, 1) * n) + s] = -h7;
        p[(idx_im(2, 1) * n) + s] = h6;
    }
}

// K_per_link = (1/2) Tr(P† P) = (1/2)·sum_18 P_storage[k]². Returns sum over
// n links in this direction. Matches the SU(2) convention.
[[gnu::always_inline]] inline double kinetic_slab(double const* p, std::size_t n) noexcept {
    double k = 0.0;
    for (std::size_t s = 0; s < n; ++s) {
        double per_link = 0.0;
        for (std::size_t c = 0; c < 18; ++c) {
            double const v = p[(c * n) + s];
            per_link += v * v;
        }
        k += per_link;
    }
    return 0.5 * k;
}

}  // namespace reticolo::math::su3
