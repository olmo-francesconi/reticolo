#pragma once

#include <reticolo/core/parallel.hpp>
#include <reticolo/core/rng/philox.hpp>
#include <reticolo/math/vec_libm.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numbers>
#include <vector>

namespace reticolo::math::su3 {

// NOTE: the per-link formulas here (mul/mul_adj/adj_mul/traceless_antiherm/
// exp_su3) are copied verbatim into the device path cuda::SU3Device
// (cuda/gauge/su3_device.cuh) — the CPU side is Sleef-batched slabs, the device
// side is register-local, so they can't literally share code. If you edit the
// math below, mirror it there; test_cuda_su3 asserts device-vs-host agreement
// and will fail on drift.
//
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

// ---------- batched AoSoA 3×3 complex products -------------------------------
// Operate on `[9][B]` re/im slabs: entry (i,j) of site b lives at
// `re[(3*i)+j][b]` / `im[(3*i)+j][b]`. The innermost b-loop is stride-1
// packed data, so the compiler vectorises it at the target SIMD width with
// no cross-lane permutation. `Acc` selects `out +=` over `out =`. Out must
// not alias the inputs.

// out (+)= a · b
template <bool Acc, std::size_t B, class T>
[[gnu::always_inline]] inline void mul_3x3_batched(T (&out_re)[9][B],
                                                   T (&out_im)[9][B],
                                                   T const (&a_re)[9][B],
                                                   T const (&a_im)[9][B],
                                                   T const (&b_re)[9][B],
                                                   T const (&b_im)[9][B]) noexcept {
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            T cr[B];
            T ci[B];
            for (std::size_t b = 0; b < B; ++b) {
                cr[b] = T{0};
                ci[b] = T{0};
            }
            for (std::size_t k = 0; k < 3; ++k) {
                std::size_t const ka = (3 * i) + k;
                std::size_t const kb = (3 * k) + j;
                for (std::size_t b = 0; b < B; ++b) {
                    cr[b] += (a_re[ka][b] * b_re[kb][b]) - (a_im[ka][b] * b_im[kb][b]);
                    ci[b] += (a_re[ka][b] * b_im[kb][b]) + (a_im[ka][b] * b_re[kb][b]);
                }
            }
            std::size_t const out_k = (3 * i) + j;
            for (std::size_t b = 0; b < B; ++b) {
                if constexpr (Acc) {
                    out_re[out_k][b] += cr[b];
                    out_im[out_k][b] += ci[b];
                } else {
                    out_re[out_k][b] = cr[b];
                    out_im[out_k][b] = ci[b];
                }
            }
        }
    }
}

// out (+)= a · b†   →   out_{ij} = sum_k a_{ik} · conj(b_{jk})
template <bool Acc, std::size_t B, class T>
[[gnu::always_inline]] inline void mul_adj_3x3_batched(T (&out_re)[9][B],
                                                       T (&out_im)[9][B],
                                                       T const (&a_re)[9][B],
                                                       T const (&a_im)[9][B],
                                                       T const (&b_re)[9][B],
                                                       T const (&b_im)[9][B]) noexcept {
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            T cr[B];
            T ci[B];
            for (std::size_t b = 0; b < B; ++b) {
                cr[b] = T{0};
                ci[b] = T{0};
            }
            for (std::size_t k = 0; k < 3; ++k) {
                std::size_t const ka = (3 * i) + k;
                std::size_t const kb = (3 * j) + k;
                for (std::size_t b = 0; b < B; ++b) {
                    cr[b] += (a_re[ka][b] * b_re[kb][b]) + (a_im[ka][b] * b_im[kb][b]);
                    ci[b] += (a_im[ka][b] * b_re[kb][b]) - (a_re[ka][b] * b_im[kb][b]);
                }
            }
            std::size_t const out_k = (3 * i) + j;
            for (std::size_t b = 0; b < B; ++b) {
                if constexpr (Acc) {
                    out_re[out_k][b] += cr[b];
                    out_im[out_k][b] += ci[b];
                } else {
                    out_re[out_k][b] = cr[b];
                    out_im[out_k][b] = ci[b];
                }
            }
        }
    }
}

// out (+)= a† · b   →   out_{ij} = sum_k conj(a_{ki}) · b_{kj}
template <bool Acc, std::size_t B, class T>
[[gnu::always_inline]] inline void adj_mul_3x3_batched(T (&out_re)[9][B],
                                                       T (&out_im)[9][B],
                                                       T const (&a_re)[9][B],
                                                       T const (&a_im)[9][B],
                                                       T const (&b_re)[9][B],
                                                       T const (&b_im)[9][B]) noexcept {
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            T cr[B];
            T ci[B];
            for (std::size_t b = 0; b < B; ++b) {
                cr[b] = T{0};
                ci[b] = T{0};
            }
            for (std::size_t k = 0; k < 3; ++k) {
                std::size_t const ka = (3 * k) + i;
                std::size_t const kb = (3 * k) + j;
                for (std::size_t b = 0; b < B; ++b) {
                    cr[b] += (a_re[ka][b] * b_re[kb][b]) + (a_im[ka][b] * b_im[kb][b]);
                    ci[b] += (a_re[ka][b] * b_im[kb][b]) - (a_im[ka][b] * b_re[kb][b]);
                }
            }
            std::size_t const out_k = (3 * i) + j;
            for (std::size_t b = 0; b < B; ++b) {
                if constexpr (Acc) {
                    out_re[out_k][b] += cr[b];
                    out_im[out_k][b] += ci[b];
                } else {
                    out_re[out_k][b] = cr[b];
                    out_im[out_k][b] = ci[b];
                }
            }
        }
    }
}

// Batched TA(M): diagonal = i·(Im_{ii} − Tr/3); off-diag (i<j) is the
// anti-hermitian completion. Same math as traceless_antiherm_3x3.
template <std::size_t B, class T>
[[gnu::always_inline]] inline void traceless_antiherm_3x3_batched(T (&ta_re)[9][B],
                                                                  T (&ta_im)[9][B],
                                                                  T const (&in_re)[9][B],
                                                                  T const (&in_im)[9][B]) noexcept {
    for (std::size_t b = 0; b < B; ++b) {
        T const t_over_3 = (in_im[0][b] + in_im[4][b] + in_im[8][b]) / T{3};
        ta_re[0][b]      = T{0};
        ta_im[0][b]      = in_im[0][b] - t_over_3;
        ta_re[4][b]      = T{0};
        ta_im[4][b]      = in_im[4][b] - t_over_3;
        ta_re[8][b]      = T{0};
        ta_im[8][b]      = in_im[8][b] - t_over_3;
        T const re01     = T{0.5} * (in_re[1][b] - in_re[3][b]);
        T const im01     = T{0.5} * (in_im[1][b] + in_im[3][b]);
        ta_re[1][b]      = re01;
        ta_im[1][b]      = im01;
        ta_re[3][b]      = -re01;
        ta_im[3][b]      = im01;
        T const re02     = T{0.5} * (in_re[2][b] - in_re[6][b]);
        T const im02     = T{0.5} * (in_im[2][b] + in_im[6][b]);
        ta_re[2][b]      = re02;
        ta_im[2][b]      = im02;
        ta_re[6][b]      = -re02;
        ta_im[6][b]      = im02;
        T const re12     = T{0.5} * (in_re[5][b] - in_re[7][b]);
        T const im12     = T{0.5} * (in_im[5][b] + in_im[7][b]);
        ta_re[5][b]      = re12;
        ta_im[5][b]      = im12;
        ta_re[7][b]      = -re12;
        ta_im[7][b]      = im12;
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
    for (double const k : r0) {
        n0_sq += k * k;
    }
    double const inv_n0 = 1.0 / std::sqrt(n0_sq);
    for (double& k : r0) {
        k *= inv_n0;
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
    for (double const k : r1) {
        n1_sq += k * k;
    }
    double const inv_n1 = 1.0 / std::sqrt(n1_sq);
    for (double& k : r1) {
        k *= inv_n1;
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

namespace impl {

// Cayley-Hamilton coefficient pass of expi_lmul_slab. Per site: branchless
// straight-line block (selects only), so the loop vectorises across sites.
// Small-c1 lanes blend to the Taylor coefficients, which fit the same
// f₀·I + f₁·Q + f₂·Q² form via Q³ = c1·Q + c0·I —
//   f₀ = 1 − i·c0/6,  f₁ = i·(1 − c1/6),  f₂ = −1/2.
// The main-path values are still computed on those lanes (inv_den may
// overflow to inf and produce NaN); the select discards them. Split out as a
// function so the slab pointers carry __restrict: the f outputs reuse dead
// scratch slabs and the vectoriser otherwise has to prove all stores/loads
// disjoint at runtime.
inline void ch_coeff_pass_(double* __restrict f0_re,
                           double* __restrict f0_im,
                           double* __restrict f1_re,
                           double* __restrict f1_im,
                           double* __restrict f2_re,
                           double* __restrict f2_im,
                           double const* __restrict u_buf,
                           double const* __restrict w_buf,
                           double const* __restrict cu_buf,
                           double const* __restrict su_buf,
                           double const* __restrict cw_buf,
                           double const* __restrict sw_buf,
                           double const* __restrict c0_buf,
                           double const* __restrict c1_buf,
                           std::size_t n) noexcept {
    constexpr double k_small_c1 = 1.0e-8;
    constexpr double k_eps_w    = 1.0e-10;
    constexpr double k_inv_6    = 1.0 / 6.0;
    // The NEON cost model rejects this loop (two fdivs); measured, the
    // vector version still wins — the ~60 mul/add ops dominate and the
    // divides pipeline. GCC's model vectorises it unaided.
#if defined(__clang__)
    #pragma clang loop vectorize(enable)
#endif
    for (std::size_t s = 0; s < n; ++s) {
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
        bool const small     = c1_buf[s] < k_small_c1;
        f0_re[s]             = small ? 1.0 : h0_re * inv_den;
        f0_im[s]             = small ? -c0_buf[s] * k_inv_6 : h0_im * inv_den;
        f1_re[s]             = small ? 0.0 : h1_re * inv_den;
        f1_im[s]             = small ? 1.0 - (c1_buf[s] * k_inv_6) : h1_im * inv_den;
        f2_re[s]             = small ? -0.5 : h2_re * inv_den;
        f2_im[s]             = small ? 0.0 : h2_im * inv_den;
    }
}

}  // namespace impl

// In-place U ← exp(dt·P) · U, slab edition.
//
// The per-site Cayley-Hamilton exp has 1 acos + 5 sincos calls per link plus
// 2 sqrt. Batching them across all slab sites via Sleef cuts the
// transcendental cost ~2× on AArch64 / AVX2. The structure is multi-pass,
// every pass site-parallel (stride-1 loads, no loop-carried state):
//
//  pass 1 (vector):   c1 = ½·dt²·Σ p², c0 = det Q (straight-line, no matrix
//                     temporaries — Q is a signed permutation of dt·P, and
//                     for traceless hermitian Q, Tr(Q³)/3 = det Q), clamped
//                     ratio = c0/c0_max, √(c1/3), √c1.
//  pass 2 (vector):   theta = acos(ratio) via acos_batch.
//  pass 3 (vector):   build theta/3 buffer; sincos_batch → (s_t3, c_t3).
//  pass 3.5 (vector): u = √(c1/3)·c_t3, w = √c1·s_t3.
//  pass 4 (vector):   sincos_batch(u) → (su, cu), sincos_batch(w) → (sw, cw).
//  pass 4.5 (vector): Cayley-Hamilton coefficients f₀, f₁, f₂. Branchless:
//                     small-c1 sites blend to the Taylor coefficients, which
//                     fit the same form via Q³ = c1·Q + c0·I —
//                     f₀ = 1 − i·c0/6, f₁ = i·(1 − c1/6), f₂ = −1/2.
//  pass 5 (batched):  V = f₀·I + f₁·Q + f₂·Q² and U ← V·U in AoSoA batches
//                     of B sites (separate re/im [9][B] slabs, batched 3×3
//                     products) so the matrix work vectorises across sites.
//
// 2u-trig is computed inline as c2u = cu² − su² and s2u = 2·cu·su (no extra
// transcendentals). ξ = sin(w)/w uses a branchless small-w guard.
// Templated on the field precision T: the link storage U and momentum P are
// loaded/stored as T (float halves the bandwidth), but the Cayley-Hamilton
// exponential itself is evaluated in double — the acos near ±1, the 9u²−w²
// denominator and the Taylor blend are precision-delicate. The site batching
// recovers the SIMD width the double arithmetic allows.
// Range worker: apply U ← exp(i·dt·P)·U to the `cnt` links [base, base+cnt) of
// one direction slab whose per-component stride is `stride` (= full nsites).
// Pure — no threading. Scratch is sized to the chunk and indexed locally in
// [0, cnt); the link/momentum loads and stores use the global site index
// `base + s` with the full stride. When the caller keeps chunk boundaries
// k_b-aligned, the batched pass-5 groupings match the whole-slab sweep exactly,
// so the result is bit-identical for any partition.
template <class T>
inline void expi_lmul_range(
    T* u, T const* p, double dt, std::size_t stride, std::size_t base, std::size_t cnt) noexcept {
    thread_local std::vector<double> scratch;
    constexpr std::size_t k_slabs = 14;
    reticolo::exec::thread_scratch(scratch, k_slabs * cnt);
    double* const ratio_buf = scratch.data() + (0 * cnt);  // pass1→pass2
    double* const theta_buf = scratch.data() + (1 * cnt);  // pass2→pass3 prep
    double* const t3_buf    = scratch.data() + (2 * cnt);  // pass3 input
    double* const sqrt_c1_3 = scratch.data() + (3 * cnt);
    double* const sqrt_c1   = scratch.data() + (4 * cnt);
    double* const c0_buf    = scratch.data() + (5 * cnt);
    double* const u_buf     = scratch.data() + (6 * cnt);
    double* const w_buf     = scratch.data() + (7 * cnt);
    double* const cu_buf    = scratch.data() + (8 * cnt);
    double* const su_buf    = scratch.data() + (9 * cnt);
    double* const cw_buf    = scratch.data() + (10 * cnt);
    double* const sw_buf    = scratch.data() + (11 * cnt);
    double* const s_t3_buf  = scratch.data() + (12 * cnt);
    double* const c1_buf    = scratch.data() + (13 * cnt);

    constexpr double k_inv_3 = 1.0 / 3.0;

    // ----- Pass 1: c1, c0 = det Q, clamped ratio, √c1 forms ---------------
    // Q = -i·dt·P is a signed re/im swap of dt·P, so Σ Q² = dt²·Σ P² needs no
    // assembly, and det Q is a straight-line complex 3×3 determinant on the
    // swapped entries. Every load is stride-1 in s; the loop body is
    // branchless (clamps and the c0_max guard are selects), so this
    // vectorises across sites.
    for (std::size_t s = 0; s < cnt; ++s) {
        std::size_t const g = base + s;
        double q_re[9];
        double q_im[9];
        for (std::size_t k = 0; k < 9; ++k) {
            q_re[k] = dt * static_cast<double>(p[(((2 * k) + 1) * stride) + g]);
            q_im[k] = -dt * static_cast<double>(p[((2 * k) * stride) + g]);
        }
        double sum_sq = 0.0;
        for (std::size_t k = 0; k < 9; ++k) {
            sum_sq += (q_re[k] * q_re[k]) + (q_im[k] * q_im[k]);
        }
        double const c1 = 0.5 * sum_sq;
        // det Q = Q00·(Q11 Q22 − Q12 Q21) − Q01·(Q10 Q22 − Q12 Q20)
        //       + Q02·(Q10 Q21 − Q11 Q20); real for hermitian Q.
        auto cmul_re = [](double ar, double ai, double br, double bi) noexcept {
            return (ar * br) - (ai * bi);
        };
        auto cmul_im = [](double ar, double ai, double br, double bi) noexcept {
            return (ar * bi) + (ai * br);
        };
        double const m0_re = cmul_re(q_re[4], q_im[4], q_re[8], q_im[8]) -
                             cmul_re(q_re[5], q_im[5], q_re[7], q_im[7]);
        double const m0_im = cmul_im(q_re[4], q_im[4], q_re[8], q_im[8]) -
                             cmul_im(q_re[5], q_im[5], q_re[7], q_im[7]);
        double const m1_re = cmul_re(q_re[3], q_im[3], q_re[8], q_im[8]) -
                             cmul_re(q_re[5], q_im[5], q_re[6], q_im[6]);
        double const m1_im = cmul_im(q_re[3], q_im[3], q_re[8], q_im[8]) -
                             cmul_im(q_re[5], q_im[5], q_re[6], q_im[6]);
        double const m2_re = cmul_re(q_re[3], q_im[3], q_re[7], q_im[7]) -
                             cmul_re(q_re[4], q_im[4], q_re[6], q_im[6]);
        double const m2_im = cmul_im(q_re[3], q_im[3], q_re[7], q_im[7]) -
                             cmul_im(q_re[4], q_im[4], q_re[6], q_im[6]);
        double const c0    = cmul_re(q_re[0], q_im[0], m0_re, m0_im) -
                             cmul_re(q_re[1], q_im[1], m1_re, m1_im) +
                             cmul_re(q_re[2], q_im[2], m2_re, m2_im);

        double const c1_over_3 = c1 * k_inv_3;
        double const sc13      = std::sqrt(c1_over_3);
        double const sc1       = std::sqrt(c1);
        double const c0_max    = 2.0 * c1_over_3 * sc13;
        double const ratio     = (c0_max > 0.0) ? std::max(-1.0, std::min(1.0, c0 / c0_max)) : 0.0;
        ratio_buf[s]           = ratio;
        sqrt_c1_3[s]           = sc13;
        sqrt_c1[s]             = sc1;
        c0_buf[s]              = c0;
        c1_buf[s]              = c1;
    }

    // ----- Pass 2: theta = acos(ratio) -----------------------------------
    reticolo::math::acos_batch(theta_buf, ratio_buf, cnt);

    // ----- Pass 3: theta/3 → sincos --------------------------------------
    for (std::size_t s = 0; s < cnt; ++s) {
        t3_buf[s] = theta_buf[s] * k_inv_3;
    }
    reticolo::math::sincos_batch(s_t3_buf, /*cos*/ theta_buf, t3_buf, cnt);
    //   ↑ reuse theta_buf for cos(theta/3) — its acos contents are no longer
    //     needed. Calling convention of sincos_batch is (sin_dst, cos_dst, src, n).

    // ----- Pass 3.5: u, w ------------------------------------------------
    for (std::size_t s = 0; s < cnt; ++s) {
        double const c_t3 = theta_buf[s];
        double const s_t3 = s_t3_buf[s];
        u_buf[s]          = sqrt_c1_3[s] * c_t3;
        w_buf[s]          = sqrt_c1[s] * s_t3;
    }

    // ----- Pass 4: sincos(u), sincos(w) ----------------------------------
    reticolo::math::sincos_batch(su_buf, cu_buf, u_buf, cnt);
    reticolo::math::sincos_batch(sw_buf, cw_buf, w_buf, cnt);

    // ----- Pass 4.5: Cayley-Hamilton coefficients f₀, f₁, f₂ --------------
    // The f slabs reuse buffers that are dead after pass 3.5; the kernel is
    // split out (impl::ch_coeff_pass_) so its parameters carry __restrict —
    // without it the vectoriser must prove the 6 stores and 8 loads disjoint
    // at runtime and its cost model gives up.
    double* const f0_re_buf = ratio_buf;
    double* const f0_im_buf = theta_buf;
    double* const f1_re_buf = t3_buf;
    double* const f1_im_buf = sqrt_c1_3;
    double* const f2_re_buf = sqrt_c1;
    double* const f2_im_buf = s_t3_buf;
    impl::ch_coeff_pass_(f0_re_buf,
                         f0_im_buf,
                         f1_re_buf,
                         f1_im_buf,
                         f2_re_buf,
                         f2_im_buf,
                         u_buf,
                         w_buf,
                         cu_buf,
                         su_buf,
                         cw_buf,
                         sw_buf,
                         c0_buf,
                         c1_buf,
                         cnt);

    // ----- Pass 5: batched V = f₀·I + f₁·Q + f₂·Q², U ← V·U ---------------
    // AoSoA batches of B sites: q/q²/u slabs split into [9][B] re/im scratch
    // so the matrix products vectorise across sites (stride-1 in b).
    constexpr std::size_t k_b   = 8;
    std::size_t const tail_base = (cnt / k_b) * k_b;
    for (std::size_t s0 = 0; s0 < tail_base; s0 += k_b) {
        std::size_t const g0 = base + s0;
        double q_re[9][k_b];
        double q_im[9][k_b];
        double uo_re[9][k_b];
        double uo_im[9][k_b];
        for (std::size_t k = 0; k < 9; ++k) {
            T const* p_re = p + ((2 * k) * stride) + g0;
            T const* p_im = p + (((2 * k) + 1) * stride) + g0;
            T const* u_re = u + ((2 * k) * stride) + g0;
            T const* u_im = u + (((2 * k) + 1) * stride) + g0;
            for (std::size_t b = 0; b < k_b; ++b) {
                q_re[k][b]  = dt * static_cast<double>(p_im[b]);
                q_im[k][b]  = -dt * static_cast<double>(p_re[b]);
                uo_re[k][b] = static_cast<double>(u_re[b]);
                uo_im[k][b] = static_cast<double>(u_im[b]);
            }
        }
        double q2_re[9][k_b];
        double q2_im[9][k_b];
        mul_3x3_batched<false>(q2_re, q2_im, q_re, q_im, q_re, q_im);

        double v_re[9][k_b];
        double v_im[9][k_b];
        for (std::size_t k = 0; k < 9; ++k) {
            for (std::size_t b = 0; b < k_b; ++b) {
                double const f1r = f1_re_buf[s0 + b];
                double const f1i = f1_im_buf[s0 + b];
                double const f2r = f2_re_buf[s0 + b];
                double const f2i = f2_im_buf[s0 + b];
                v_re[k][b]       = ((f1r * q_re[k][b]) - (f1i * q_im[k][b])) +
                                   ((f2r * q2_re[k][b]) - (f2i * q2_im[k][b]));
                v_im[k][b]       = ((f1r * q_im[k][b]) + (f1i * q_re[k][b])) +
                                   ((f2r * q2_im[k][b]) + (f2i * q2_re[k][b]));
            }
        }
        for (std::size_t k = 0; k < 9; k += 4) {  // diagonal entries 0, 4, 8
            for (std::size_t b = 0; b < k_b; ++b) {
                v_re[k][b] += f0_re_buf[s0 + b];
                v_im[k][b] += f0_im_buf[s0 + b];
            }
        }

        double o_re[9][k_b];
        double o_im[9][k_b];
        mul_3x3_batched<false>(o_re, o_im, v_re, v_im, uo_re, uo_im);
        for (std::size_t k = 0; k < 9; ++k) {
            T* u_re = u + ((2 * k) * stride) + g0;
            T* u_im = u + (((2 * k) + 1) * stride) + g0;
            for (std::size_t b = 0; b < k_b; ++b) {
                u_re[b] = static_cast<T>(o_re[k][b]);
                u_im[b] = static_cast<T>(o_im[k][b]);
            }
        }
    }

    // Scalar tail for the last cnt % k_b sites, same math per site.
    for (std::size_t s = tail_base; s < cnt; ++s) {
        std::size_t const g = base + s;
        double q[18];
        for (std::size_t k = 0; k < 9; ++k) {
            q[2 * k]       = dt * static_cast<double>(p[(((2 * k) + 1) * stride) + g]);
            q[(2 * k) + 1] = -dt * static_cast<double>(p[((2 * k) * stride) + g]);
        }
        double q2[18];
        mul_3x3(q2, q, q);
        double const f0r = f0_re_buf[s];
        double const f0i = f0_im_buf[s];
        double const f1r = f1_re_buf[s];
        double const f1i = f1_im_buf[s];
        double const f2r = f2_re_buf[s];
        double const f2i = f2_im_buf[s];
        double v[18];
        for (std::size_t k = 0; k < 9; ++k) {
            std::size_t const kr = 2 * k;
            std::size_t const ki = kr + 1;
            v[kr] = ((f1r * q[kr]) - (f1i * q[ki])) + ((f2r * q2[kr]) - (f2i * q2[ki]));
            v[ki] = ((f1r * q[ki]) + (f1i * q[kr])) + ((f2r * q2[ki]) + (f2i * q2[kr]));
        }
        v[idx_re(0, 0)] += f0r;
        v[idx_im(0, 0)] += f0i;
        v[idx_re(1, 1)] += f0r;
        v[idx_im(1, 1)] += f0i;
        v[idx_re(2, 2)] += f0r;
        v[idx_im(2, 2)] += f0i;

        double u_old[18];
        for (std::size_t k = 0; k < 18; ++k) {
            u_old[k] = static_cast<double>(u[(k * stride) + g]);
        }
        double o_s[18];
        mul_3x3(o_s, v, u_old);
        for (std::size_t k = 0; k < 18; ++k) {
            u[(k * stride) + g] = static_cast<T>(o_s[k]);
        }
    }
}

// U ← exp(i·dt·P)·U over a full direction slab of `n` links (serial). The
// parallel drift partition lives in the integrator op layer (integ_ops.hpp),
// which calls `expi_lmul_range` per thread-chunk; this stays pure math.
template <class T>
inline void expi_lmul_slab(T* u, T const* p, double dt, std::size_t n) noexcept {
    expi_lmul_range(u, p, dt, n, 0, n);
}

// Padded form: `count` links with component stride `stride` (≥ count). Both u
// and p must share this stride. Packed (stride == count) matches the 4-arg form.
template <class T>
inline void
expi_lmul_slab(T* u, T const* p, double dt, std::size_t stride, std::size_t count) noexcept {
    expi_lmul_range(u, p, dt, stride, 0, count);
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
// (stride, count) form: write `count` links' algebra with component stride
// `stride` (≥ count, padded ≥ nsites for cache-friendly gauge storage). The
// draw order is a pure function of `count` (8·count normals, count-major), so
// for stride == count this is bit-identical to the historic packed layout —
// the 2-arg overload below forwards to exactly that.
template <class T, class Rng>
[[gnu::always_inline]] inline void
sample_algebra_slab(T* p, Rng& rng, std::size_t stride, std::size_t count) noexcept {
    constexpr double k_inv_sqrt2 = std::numbers::sqrt2 / 2.0;
    constexpr double k_inv_sqrt3 = std::numbers::inv_sqrt3;
    // Pre-fill 8·count independent N(0, 1/√2) draws into a thread-local buffer
    // then scatter — same shape as SU(2)::sample_algebra_slab.
    thread_local std::vector<double> h_buf;
    double* const h = reticolo::exec::thread_scratch(h_buf, 8 * count);
    rng.normal_fill(h, 8 * count);
    double const* const h1_arr = h + (0 * count);
    double const* const h2_arr = h + (1 * count);
    double const* const h3_arr = h + (2 * count);
    double const* const h4_arr = h + (3 * count);
    double const* const h5_arr = h + (4 * count);
    double const* const h6_arr = h + (5 * count);
    double const* const h7_arr = h + (6 * count);
    double const* const h8_arr = h + (7 * count);
    for (std::size_t s = 0; s < count; ++s) {
        T const h1            = static_cast<T>(h1_arr[s] * k_inv_sqrt2);
        T const h2            = static_cast<T>(h2_arr[s] * k_inv_sqrt2);
        T const h3            = static_cast<T>(h3_arr[s] * k_inv_sqrt2);
        T const h4            = static_cast<T>(h4_arr[s] * k_inv_sqrt2);
        T const h5            = static_cast<T>(h5_arr[s] * k_inv_sqrt2);
        T const h6            = static_cast<T>(h6_arr[s] * k_inv_sqrt2);
        T const h7            = static_cast<T>(h7_arr[s] * k_inv_sqrt2);
        T const h8            = static_cast<T>(h8_arr[s] * k_inv_sqrt2);
        T const h8_over_sqrt3 = static_cast<T>(static_cast<double>(h8) * k_inv_sqrt3);
        // Diagonal: 00 = i·(h3 + h8/√3), 11 = i·(-h3 + h8/√3), 22 = i·(-2 h8/√3)
        p[(idx_re(0, 0) * stride) + s] = T{0};
        p[(idx_im(0, 0) * stride) + s] = h3 + h8_over_sqrt3;
        p[(idx_re(1, 1) * stride) + s] = T{0};
        p[(idx_im(1, 1) * stride) + s] = -h3 + h8_over_sqrt3;
        p[(idx_re(2, 2) * stride) + s] = T{0};
        p[(idx_im(2, 2) * stride) + s] = T{-2} * h8_over_sqrt3;
        // Off-diagonal: P_{ij} = h_re + i·h_im, P_{ji} = -h_re + i·h_im.
        p[(idx_re(0, 1) * stride) + s] = h2;
        p[(idx_im(0, 1) * stride) + s] = h1;
        p[(idx_re(1, 0) * stride) + s] = -h2;
        p[(idx_im(1, 0) * stride) + s] = h1;
        p[(idx_re(0, 2) * stride) + s] = h5;
        p[(idx_im(0, 2) * stride) + s] = h4;
        p[(idx_re(2, 0) * stride) + s] = -h5;
        p[(idx_im(2, 0) * stride) + s] = h4;
        p[(idx_re(1, 2) * stride) + s] = h7;
        p[(idx_im(1, 2) * stride) + s] = h6;
        p[(idx_re(2, 1) * stride) + s] = -h7;
        p[(idx_im(2, 1) * stride) + s] = h6;
    }
}

// Packed convenience: stride == count == n (component slab of exactly n links).
template <class T, class Rng>
[[gnu::always_inline]] inline void sample_algebra_slab(T* p, Rng& rng, std::size_t n) noexcept {
    sample_algebra_slab(p, rng, n, n);
}

// Parallel counter-based momentum sampler over links [base, base+cnt) of one
// direction `mu` (component stride `stride`). Each site's 8 Gell-Mann coordinates
// come from Philox keyed by (key, mu, site) — a pure function of the site index,
// so the draw worksplits, is identical for any thread count, and is bit-exact on
// resume (the caller draws `key` once per trajectory from its RNG). The ½√2 / √3
// basis scaling and 18-slot packing are identical to sample_algebra_slab; only
// the entropy source changes. Opt-in SU(3) replacement for the serial FastRng
// fill — unifies the CPU sampler with the counter-based device path.
template <class T>
inline void sample_algebra_philox_range(T* p,
                                        std::uint64_t key,
                                        std::uint64_t mu,
                                        std::size_t stride,
                                        std::size_t base,
                                        std::size_t cnt) noexcept {
    constexpr double k_inv_sqrt2 = std::numbers::sqrt2 / 2.0;
    constexpr double k_inv_sqrt3 = std::numbers::inv_sqrt3;
    std::size_t const end        = base + cnt;
    for (std::size_t s = base; s < end; ++s) {
        double g[8];
        for (std::size_t j = 0; j < 4; ++j) {
            reticolo::philox_normal2(key, mu, (s * 4) + j, g[2 * j], g[(2 * j) + 1]);
        }
        T const h1                     = static_cast<T>(g[0] * k_inv_sqrt2);
        T const h2                     = static_cast<T>(g[1] * k_inv_sqrt2);
        T const h3                     = static_cast<T>(g[2] * k_inv_sqrt2);
        T const h4                     = static_cast<T>(g[3] * k_inv_sqrt2);
        T const h5                     = static_cast<T>(g[4] * k_inv_sqrt2);
        T const h6                     = static_cast<T>(g[5] * k_inv_sqrt2);
        T const h7                     = static_cast<T>(g[6] * k_inv_sqrt2);
        T const h8                     = static_cast<T>(g[7] * k_inv_sqrt2);
        T const h8_over_sqrt3          = static_cast<T>(static_cast<double>(h8) * k_inv_sqrt3);
        p[(idx_re(0, 0) * stride) + s] = T{0};
        p[(idx_im(0, 0) * stride) + s] = h3 + h8_over_sqrt3;
        p[(idx_re(1, 1) * stride) + s] = T{0};
        p[(idx_im(1, 1) * stride) + s] = -h3 + h8_over_sqrt3;
        p[(idx_re(2, 2) * stride) + s] = T{0};
        p[(idx_im(2, 2) * stride) + s] = T{-2} * h8_over_sqrt3;
        p[(idx_re(0, 1) * stride) + s] = h2;
        p[(idx_im(0, 1) * stride) + s] = h1;
        p[(idx_re(1, 0) * stride) + s] = -h2;
        p[(idx_im(1, 0) * stride) + s] = h1;
        p[(idx_re(0, 2) * stride) + s] = h5;
        p[(idx_im(0, 2) * stride) + s] = h4;
        p[(idx_re(2, 0) * stride) + s] = -h5;
        p[(idx_im(2, 0) * stride) + s] = h4;
        p[(idx_re(1, 2) * stride) + s] = h7;
        p[(idx_im(1, 2) * stride) + s] = h6;
        p[(idx_re(2, 1) * stride) + s] = -h7;
        p[(idx_im(2, 1) * stride) + s] = h6;
    }
}

// Raw Σ_{s ∈ [base, base+cnt)} Σ_c p[c·stride + s]²  (no ½). Pure per-range
// reduction worker — the ½ and the cross-range fold are applied by the caller.
// Blocked: T-precision lane accumulators over k_b sites folded to double once per
// block, so a fixed k_b-block partition (see parallel_reduce) is thread-invariant.
template <class T>
[[gnu::always_inline]] inline double
kinetic_range(T const* p, std::size_t stride, std::size_t base, std::size_t cnt) noexcept {
    constexpr std::size_t k_b = 8;
    std::size_t const n_full  = (cnt / k_b) * k_b;
    std::size_t const tail_lo = base + n_full;
    std::size_t const end     = base + cnt;
    double k                  = 0.0;
    for (std::size_t s0 = base; s0 < tail_lo; s0 += k_b) {
        T acc[k_b];
        for (auto& b : acc) {
            b = T{0};
        }
        for (std::size_t c = 0; c < 18; ++c) {
            T const* row = p + (c * stride) + s0;
            for (std::size_t b = 0; b < k_b; ++b) {
                acc[b] += row[b] * row[b];
            }
        }
        double blk = 0.0;
        for (auto& b : acc) {
            blk += static_cast<double>(b);
        }
        k += blk;
    }
    for (std::size_t s = tail_lo; s < end; ++s) {
        T per_link = T{0};
        for (std::size_t c = 0; c < 18; ++c) {
            T const v = p[(c * stride) + s];
            per_link += v * v;
        }
        k += static_cast<double>(per_link);
    }
    return k;
}

// K_per_link = (1/2) Tr(P† P) = (1/2)·sum_18 P_storage[k]². Returns sum over
// n links in this direction. Matches the SU(2) convention.
template <class T>
[[gnu::always_inline]] inline double
kinetic_slab(T const* p, std::size_t stride, std::size_t count) noexcept {
    return 0.5 * kinetic_range(p, stride, 0, count);
}

template <class T>
[[gnu::always_inline]] inline double kinetic_slab(T const* p, std::size_t n) noexcept {
    return 0.5 * kinetic_range(p, n, 0, n);
}

}  // namespace reticolo::math::su3
