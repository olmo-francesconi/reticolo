#pragma once

#include <reticolo/core/sys/hd.hpp>

#include <cmath>

// Single-sourced per-link SU(2) matrix algebra — shared verbatim between the
// CPU path (math::su2, the Sleef-batched slab loops in su2_ops.hpp) and the
// CUDA device path (cuda::SU2Device, cuda/gauge/su2_device.cuh, which forwards
// to these functions). RETICOLO_HD so the exact same body compiles for both
// backends — there is no longer a second hand-copied implementation to drift.
//
// Storage layout (one link element, 8 real doubles):
//
//     k=0 : Re U_{00}    k=1 : Im U_{00}
//     k=2 : Re U_{01}    k=3 : Im U_{01}
//     k=4 : Re U_{10}    k=5 : Im U_{10}
//     k=6 : Re U_{11}    k=7 : Im U_{11}

namespace reticolo::math::su2 {

// ---------- per-site complex-multiply primitives -----------------------------

// (acc_re, acc_im) += (ar+iai) * (br+ibi)
template <class T>
[[gnu::always_inline]] RETICOLO_HD inline void
cmul_acc(T& acc_re, T& acc_im, T ar, T ai, T br, T bi) noexcept {
    acc_re += (ar * br) - (ai * bi);
    acc_im += (ar * bi) + (ai * br);
}

// (acc_re, acc_im) += (ar+iai) * conj(br+ibi) = (ar+iai)(br-ibi)
[[gnu::always_inline]] RETICOLO_HD inline void cmul_acc_b_conj(
    double& acc_re, double& acc_im, double ar, double ai, double br, double bi) noexcept {
    acc_re += (ar * br) + (ai * bi);
    acc_im += (ai * br) - (ar * bi);
}

// (acc_re, acc_im) += conj(ar+iai) * (br+ibi) = (ar-iai)(br+ibi)
[[gnu::always_inline]] RETICOLO_HD inline void cmul_acc_a_conj(
    double& acc_re, double& acc_im, double ar, double ai, double br, double bi) noexcept {
    acc_re += (ar * br) + (ai * bi);
    acc_im += (ar * bi) - (ai * br);
}

// ---------- 2×2 complex matrix products on 8-real stack arrays ---------------
// Each helper takes inputs / outputs as `double[8]` in the storage layout
// above. Out must not alias the inputs. Hand-unrolled — 8 multiply-adds per
// output entry, 4 entries = 32 mul + 24 add per product.

template <class T>
[[gnu::always_inline]] RETICOLO_HD inline void mul_2x2(T* out, T const* a, T const* b) noexcept {
    // out_{00} = a_{00}·b_{00} + a_{01}·b_{10}
    out[0] = T{0};
    out[1] = T{0};
    cmul_acc(out[0], out[1], a[0], a[1], b[0], b[1]);
    cmul_acc(out[0], out[1], a[2], a[3], b[4], b[5]);
    // out_{01} = a_{00}·b_{01} + a_{01}·b_{11}
    out[2] = T{0};
    out[3] = T{0};
    cmul_acc(out[2], out[3], a[0], a[1], b[2], b[3]);
    cmul_acc(out[2], out[3], a[2], a[3], b[6], b[7]);
    // out_{10} = a_{10}·b_{00} + a_{11}·b_{10}
    out[4] = T{0};
    out[5] = T{0};
    cmul_acc(out[4], out[5], a[4], a[5], b[0], b[1]);
    cmul_acc(out[4], out[5], a[6], a[7], b[4], b[5]);
    // out_{11} = a_{10}·b_{01} + a_{11}·b_{11}
    out[6] = T{0};
    out[7] = T{0};
    cmul_acc(out[6], out[7], a[4], a[5], b[2], b[3]);
    cmul_acc(out[6], out[7], a[6], a[7], b[6], b[7]);
}

// out = a · b†
[[gnu::always_inline]] RETICOLO_HD inline void
mul_adj_2x2(double* out, double const* a, double const* b) noexcept {
    // out_{ij} = sum_k a_{ik} · conj(b_{jk})
    out[0] = 0.0;
    out[1] = 0.0;
    cmul_acc_b_conj(out[0], out[1], a[0], a[1], b[0], b[1]);
    cmul_acc_b_conj(out[0], out[1], a[2], a[3], b[2], b[3]);

    out[2] = 0.0;
    out[3] = 0.0;
    cmul_acc_b_conj(out[2], out[3], a[0], a[1], b[4], b[5]);
    cmul_acc_b_conj(out[2], out[3], a[2], a[3], b[6], b[7]);

    out[4] = 0.0;
    out[5] = 0.0;
    cmul_acc_b_conj(out[4], out[5], a[4], a[5], b[0], b[1]);
    cmul_acc_b_conj(out[4], out[5], a[6], a[7], b[2], b[3]);

    out[6] = 0.0;
    out[7] = 0.0;
    cmul_acc_b_conj(out[6], out[7], a[4], a[5], b[4], b[5]);
    cmul_acc_b_conj(out[6], out[7], a[6], a[7], b[6], b[7]);
}

// out = a† · b
[[gnu::always_inline]] RETICOLO_HD inline void
adj_mul_2x2(double* out, double const* a, double const* b) noexcept {
    // out_{ij} = sum_k conj(a_{ki}) · b_{kj}
    out[0] = 0.0;
    out[1] = 0.0;
    cmul_acc_a_conj(out[0], out[1], a[0], a[1], b[0], b[1]);
    cmul_acc_a_conj(out[0], out[1], a[4], a[5], b[4], b[5]);

    out[2] = 0.0;
    out[3] = 0.0;
    cmul_acc_a_conj(out[2], out[3], a[0], a[1], b[2], b[3]);
    cmul_acc_a_conj(out[2], out[3], a[4], a[5], b[6], b[7]);

    out[4] = 0.0;
    out[5] = 0.0;
    cmul_acc_a_conj(out[4], out[5], a[2], a[3], b[0], b[1]);
    cmul_acc_a_conj(out[4], out[5], a[6], a[7], b[4], b[5]);

    out[6] = 0.0;
    out[7] = 0.0;
    cmul_acc_a_conj(out[6], out[7], a[2], a[3], b[2], b[3]);
    cmul_acc_a_conj(out[6], out[7], a[6], a[7], b[6], b[7]);
}

// ---------- traceless anti-hermitian projection (su(2) algebra) -------------
// TA(M) = (M − M†)/2 − (1/N)·Tr((M−M†)/2)·I. For SU(2) the diagonals become
// ±i·(Im M_{00} − Im M_{11})/2 and the off-diagonals are the anti-hermitian
// completion of M_{01}, M_{10} (already trace-zero after the diagonal fix).
[[gnu::always_inline]] RETICOLO_HD inline void traceless_antiherm_2x2(double* out,
                                                                      double const* in) noexcept {
    double const im00    = in[1];
    double const im11    = in[7];
    double const re01    = in[2];
    double const im01    = in[3];
    double const re10    = in[4];
    double const im10    = in[5];
    double const diag_im = 0.5 * (im00 - im11);
    out[0]               = 0.0;
    out[1]               = diag_im;
    out[2]               = 0.5 * (re01 - re10);
    out[3]               = 0.5 * (im01 + im10);
    out[4]               = -out[2];
    out[5]               = out[3];
    out[6]               = 0.0;
    out[7]               = -diag_im;
}

// ---------- closed-form group exponential ------------------------------------

// Build V = exp(dt · P) where P is anti-hermitian 2×2 (su(2) algebra element)
// in the storage convention above. Output written as 8 reals at `v`.
// Branchless small-angle handling: gamma = sin(beta)/||h|| with beta = dt·||h||.
[[gnu::always_inline]] RETICOLO_HD inline void
exp_su2(double* v, double const* p, double dt) noexcept {
    double const h3   = p[1];
    double const h2   = p[2];
    double const h1   = p[3];
    double const h_sq = (h1 * h1) + (h2 * h2) + (h3 * h3);
    double const h    = std::sqrt(h_sq);
    double const beta = dt * h;
    double const c    = std::cos(beta);
    // gamma = sin(beta)/h, well-defined at h=0 via Taylor (gamma → dt).
    // The "if h tiny" branch keeps gamma finite; when h is exactly 0 all h_a
    // are 0 so gamma·h_a = 0 in the formulas below regardless.
    double const gamma = (h > 1.0e-12) ? (std::sin(beta) / h) : (dt - ((dt * beta * beta) / 6.0));
    v[0]               = c;
    v[1]               = gamma * h3;
    v[2]               = gamma * h2;
    v[3]               = gamma * h1;
    v[4]               = -gamma * h2;
    v[5]               = gamma * h1;
    v[6]               = c;
    v[7]               = -gamma * h3;
}

}  // namespace reticolo::math::su2
