#pragma once

#include <reticolo/core/philox.hpp>
#include <reticolo/math/vec_libm.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <vector>

namespace reticolo::math::su2 {

// NOTE: the per-link formulas here are copied verbatim into the device path
// cuda::SU2Device (cuda/gauge/su2_device.cuh) — CPU side is Sleef-batched slabs,
// device side is register-local, so they can't share code. Edit one, mirror the
// other; test_cuda_su2 asserts device-vs-host agreement and fails on drift.
//
// Hand-written 2×2 complex matrix kernels for SU(2) lattice gauge fields.
//
// Storage layout (one link element, 8 real doubles):
//
//     k=0 : Re U_{00}    k=1 : Im U_{00}
//     k=2 : Re U_{01}    k=3 : Im U_{01}
//     k=4 : Re U_{10}    k=5 : Im U_{10}
//     k=6 : Re U_{11}    k=7 : Im U_{11}
//
// When `MatrixLinkLattice<SU2,T>` stores a slab of n sites for direction mu,
// component k of the link at site s lives at `mu_block_data(mu)[k*n + s]`.
//
// Per-site kernels load 4 doubles per complex matrix entry into stack
// registers, do the small matrix product or projection straight-line, and
// store back; the compiler turns the outer `for (s : n)` into an auto-
// vectorised loop because every load/store inside is stride-1 in s. No
// intrinsics, no Eigen — 2×2 is below any GEMM crossover.
//
// Algebra storage convention (anti-hermitian P with three real params h_a):
//
//     P = i·(h_1·sigma_x + h_2·sigma_y + h_3·sigma_z)
//       = [  i·h_3            h_2 + i·h_1  ]
//         [ -h_2 + i·h_1     -i·h_3        ]
//
// so on the 8-real storage:
//     h_1 = Im P_{01} (= Im P_{10})
//     h_2 = Re P_{01} (= -Re P_{10})
//     h_3 = Im P_{00} (= -Im P_{11})
//     Re P_{00} = Re P_{11} = 0
//
// Sampling: each h_a ~ N(0, 1) i.i.d. (so the kinetic energy per link is
// K = ||h||^2 = (1/2) Tr(P† P)). Force code must adopt the same convention.

// ---------- per-site complex-multiply primitives -----------------------------

// (acc_re, acc_im) += (ar+iai) * (br+ibi)
template <class T>
[[gnu::always_inline]] inline void cmul_acc(T& acc_re, T& acc_im, T ar, T ai, T br, T bi) noexcept {
    acc_re += (ar * br) - (ai * bi);
    acc_im += (ar * bi) + (ai * br);
}

// (acc_re, acc_im) += (ar+iai) * conj(br+ibi) = (ar+iai)(br-ibi)
[[gnu::always_inline]] inline void cmul_acc_b_conj(
    double& acc_re, double& acc_im, double ar, double ai, double br, double bi) noexcept {
    acc_re += (ar * br) + (ai * bi);
    acc_im += (ai * br) - (ar * bi);
}

// (acc_re, acc_im) += conj(ar+iai) * (br+ibi) = (ar-iai)(br+ibi)
[[gnu::always_inline]] inline void cmul_acc_a_conj(
    double& acc_re, double& acc_im, double ar, double ai, double br, double bi) noexcept {
    acc_re += (ar * br) + (ai * bi);
    acc_im += (ar * bi) - (ai * br);
}

// ---------- 2×2 complex matrix products on 8-real stack arrays ---------------
// Each helper takes inputs / outputs as `double[8]` in the storage layout
// above. Out must not alias the inputs. Hand-unrolled — 8 multiply-adds per
// output entry, 4 entries = 32 mul + 24 add per product.

template <class T>
[[gnu::always_inline]] inline void mul_2x2(T* out, T const* a, T const* b) noexcept {
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
[[gnu::always_inline]] inline void
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
[[gnu::always_inline]] inline void
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

// ---------- batched AoSoA 2×2 complex products -------------------------------
// Operate on `[4][B]` re/im slabs: entry (i,j) of site b lives at
// `re[(2*i)+j][b]` / `im[(2*i)+j][b]`. The innermost b-loop is stride-1
// packed data, so the compiler vectorises it at the target SIMD width.
// `Acc` selects `out +=` over `out =`. Out must not alias the inputs.

// out (+)= a · b
template <bool Acc, std::size_t B, class T>
[[gnu::always_inline]] inline void mul_2x2_batched(T (&out_re)[4][B],
                                                   T (&out_im)[4][B],
                                                   T const (&a_re)[4][B],
                                                   T const (&a_im)[4][B],
                                                   T const (&b_re)[4][B],
                                                   T const (&b_im)[4][B]) noexcept {
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            T cr[B];
            T ci[B];
            for (std::size_t b = 0; b < B; ++b) {
                cr[b] = T{0};
                ci[b] = T{0};
            }
            for (std::size_t k = 0; k < 2; ++k) {
                std::size_t const ka = (2 * i) + k;
                std::size_t const kb = (2 * k) + j;
                for (std::size_t b = 0; b < B; ++b) {
                    cr[b] += (a_re[ka][b] * b_re[kb][b]) - (a_im[ka][b] * b_im[kb][b]);
                    ci[b] += (a_re[ka][b] * b_im[kb][b]) + (a_im[ka][b] * b_re[kb][b]);
                }
            }
            std::size_t const out_k = (2 * i) + j;
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
[[gnu::always_inline]] inline void mul_adj_2x2_batched(T (&out_re)[4][B],
                                                       T (&out_im)[4][B],
                                                       T const (&a_re)[4][B],
                                                       T const (&a_im)[4][B],
                                                       T const (&b_re)[4][B],
                                                       T const (&b_im)[4][B]) noexcept {
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            T cr[B];
            T ci[B];
            for (std::size_t b = 0; b < B; ++b) {
                cr[b] = T{0};
                ci[b] = T{0};
            }
            for (std::size_t k = 0; k < 2; ++k) {
                std::size_t const ka = (2 * i) + k;
                std::size_t const kb = (2 * j) + k;
                for (std::size_t b = 0; b < B; ++b) {
                    cr[b] += (a_re[ka][b] * b_re[kb][b]) + (a_im[ka][b] * b_im[kb][b]);
                    ci[b] += (a_im[ka][b] * b_re[kb][b]) - (a_re[ka][b] * b_im[kb][b]);
                }
            }
            std::size_t const out_k = (2 * i) + j;
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
[[gnu::always_inline]] inline void adj_mul_2x2_batched(T (&out_re)[4][B],
                                                       T (&out_im)[4][B],
                                                       T const (&a_re)[4][B],
                                                       T const (&a_im)[4][B],
                                                       T const (&b_re)[4][B],
                                                       T const (&b_im)[4][B]) noexcept {
    for (std::size_t i = 0; i < 2; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            T cr[B];
            T ci[B];
            for (std::size_t b = 0; b < B; ++b) {
                cr[b] = T{0};
                ci[b] = T{0};
            }
            for (std::size_t k = 0; k < 2; ++k) {
                std::size_t const ka = (2 * k) + i;
                std::size_t const kb = (2 * k) + j;
                for (std::size_t b = 0; b < B; ++b) {
                    cr[b] += (a_re[ka][b] * b_re[kb][b]) + (a_im[ka][b] * b_im[kb][b]);
                    ci[b] += (a_re[ka][b] * b_im[kb][b]) - (a_im[ka][b] * b_re[kb][b]);
                }
            }
            std::size_t const out_k = (2 * i) + j;
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

// Batched TA(M): diagonal becomes ±i·(Im_{00} − Im_{11})/2; off-diag is the
// anti-hermitian completion. Same math as traceless_antiherm_2x2.
template <std::size_t B, class T>
[[gnu::always_inline]] inline void traceless_antiherm_2x2_batched(T (&ta_re)[4][B],
                                                                  T (&ta_im)[4][B],
                                                                  T const (&in_re)[4][B],
                                                                  T const (&in_im)[4][B]) noexcept {
    for (std::size_t b = 0; b < B; ++b) {
        T const diag_im = T{0.5} * (in_im[0][b] - in_im[3][b]);
        ta_re[0][b]     = T{0};
        ta_im[0][b]     = diag_im;
        T const re01    = T{0.5} * (in_re[1][b] - in_re[2][b]);
        T const im01    = T{0.5} * (in_im[1][b] + in_im[2][b]);
        ta_re[1][b]     = re01;
        ta_im[1][b]     = im01;
        ta_re[2][b]     = -re01;
        ta_im[2][b]     = im01;
        ta_re[3][b]     = T{0};
        ta_im[3][b]     = -diag_im;
    }
}

// ---------- traceless anti-hermitian projection (su(2) algebra) -------------
// TA(M) = (M − M†)/2 − (1/N)·Tr((M−M†)/2)·I. For SU(2) the diagonals become
// ±i·(Im M_{00} − Im M_{11})/2 and the off-diagonals are the anti-hermitian
// completion of M_{01}, M_{10} (already trace-zero after the diagonal fix).
[[gnu::always_inline]] inline void traceless_antiherm_2x2(double* out, double const* in) noexcept {
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

// ---------- closed-form group exponential and projection --------------------

// Build V = exp(dt · P) where P is anti-hermitian 2×2 (su(2) algebra element)
// in the storage convention above. Output written as 8 reals at `v`.
// Branchless small-angle handling: gamma = sin(beta)/||h|| with beta = dt·||h||.
[[gnu::always_inline]] inline void exp_su2(double* v, double const* p, double dt) noexcept {
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

// Project a near-SU(2) 2×2 matrix M = [[A,B],[C,D]] onto SU(2) via the
// quaternion average: a = (A + conj(D))/2, b = (B - conj(C))/2, then
// renormalise so |a|² + |b|² = 1 and rebuild [[a,b],[-conj(b),conj(a)]].
[[gnu::always_inline]] inline void project_su2(double* m) noexcept {
    double const a_re = 0.5 * (m[0] + m[6]);
    double const a_im = 0.5 * (m[1] - m[7]);
    double const b_re = 0.5 * (m[2] - m[4]);
    double const b_im = 0.5 * (m[3] + m[5]);
    double const n_sq = (a_re * a_re) + (a_im * a_im) + (b_re * b_re) + (b_im * b_im);
    // Degenerate input (near-zero quaternion) has no meaningful SU(2)
    // projection; fall back to the identity rather than divide by ~0. Mirrors
    // the h≈0 guard in exp_su2. Physically unreachable post-trajectory.
    if (n_sq < 1.0e-24) {
        m[0] = 1.0;
        m[1] = 0.0;
        m[2] = 0.0;
        m[3] = 0.0;
        m[4] = 0.0;
        m[5] = 0.0;
        m[6] = 1.0;
        m[7] = 0.0;
        return;
    }
    double const inv = 1.0 / std::sqrt(n_sq);
    double const ar  = a_re * inv;
    double const ai  = a_im * inv;
    double const br  = b_re * inv;
    double const bi  = b_im * inv;
    m[0]             = ar;
    m[1]             = ai;
    m[2]             = br;
    m[3]             = bi;
    m[4]             = -br;
    m[5]             = bi;
    m[6]             = ar;
    m[7]             = -ai;
}

// ---------- slab driver: walk n sites, call per-site kernel inline ----------
// All slab kernels assume the n component arrays are stride-1 in site index,
// with components separated by `n`. The compiler auto-vectorises the outer
// `for (s : n)` because every load/store is stride-1.

[[gnu::always_inline]] inline void
mul_slab(double* out, double const* a, double const* b, std::size_t n) noexcept {
    for (std::size_t s = 0; s < n; ++s) {
        double a_s[8];
        double b_s[8];
        for (std::size_t k = 0; k < 8; ++k) {
            a_s[k] = a[(k * n) + s];
            b_s[k] = b[(k * n) + s];
        }
        double o_s[8];
        mul_2x2(o_s, a_s, b_s);
        for (std::size_t k = 0; k < 8; ++k) {
            out[(k * n) + s] = o_s[k];
        }
    }
}

[[gnu::always_inline]] inline void
mul_adj_slab(double* out, double const* a, double const* b, std::size_t n) noexcept {
    for (std::size_t s = 0; s < n; ++s) {
        double a_s[8];
        double b_s[8];
        for (std::size_t k = 0; k < 8; ++k) {
            a_s[k] = a[(k * n) + s];
            b_s[k] = b[(k * n) + s];
        }
        double o_s[8];
        mul_adj_2x2(o_s, a_s, b_s);
        for (std::size_t k = 0; k < 8; ++k) {
            out[(k * n) + s] = o_s[k];
        }
    }
}

[[gnu::always_inline]] inline void
adj_mul_slab(double* out, double const* a, double const* b, std::size_t n) noexcept {
    for (std::size_t s = 0; s < n; ++s) {
        double a_s[8];
        double b_s[8];
        for (std::size_t k = 0; k < 8; ++k) {
            a_s[k] = a[(k * n) + s];
            b_s[k] = b[(k * n) + s];
        }
        double o_s[8];
        adj_mul_2x2(o_s, a_s, b_s);
        for (std::size_t k = 0; k < 8; ++k) {
            out[(k * n) + s] = o_s[k];
        }
    }
}

// In-place U ← exp(dt·P) · U.
//
// Three-pass slab kernel that hoists the two transcendental calls (sin+cos
// of β = dt·‖h‖) out of the per-site body and runs them as one Sleef
// sincos_batch over a flat β scratch — replaces 2 scalar libm calls per site
// with one vectorised call per `k_vec_width_d` sites. Pass 1 computes
// β = dt·sqrt(h₁²+h₂²+h₃²) and stashes (β, h₁, h₂, h₃, ‖h‖) per site; pass
// 2 is sincos_batch; pass 3 builds V from (c, γ·h) and multiplies into U.
// γ = sin(β)/‖h‖ has a branchless small-‖h‖ guard so the inner loop stays
// vector-friendly.
//
// Range worker: apply U ← exp(dt·P)·U to the `cnt` links [base, base+cnt) of
// one direction slab whose per-component stride is `stride` (= full nsites).
// Pure — no threading. Scratch is sized to the chunk and indexed locally in
// [0, cnt); the link/momentum loads and stores use the global site index
// `base + s` with the full stride. When the caller keeps chunk boundaries
// aligned, the result is bit-identical for any partition.
template <class T>
[[gnu::always_inline]] inline void expi_lmul_range(
    T* u, T const* p, double dt, std::size_t stride, std::size_t base, std::size_t cnt) noexcept {
    // Scratch is in the field's scalar type T, so the whole kernel — including
    // the sincos transcendentals (math::sincos_batch is overloaded for float
    // and double) — runs at the lattice precision and lane count. β, ‖h‖, the
    // V matrix and the U ← V·U product are all T; for float that is 4-wide.
    thread_local std::vector<T> scratch;
    if (scratch.size() < 7 * cnt) {
        scratch.resize(7 * cnt);
    }
    T* const beta_buf = scratch.data();
    T* const h_buf    = scratch.data() + cnt;
    T* const h1_buf   = scratch.data() + (2 * cnt);
    T* const h2_buf   = scratch.data() + (3 * cnt);
    T* const h3_buf   = scratch.data() + (4 * cnt);
    T* const sin_buf  = scratch.data() + (5 * cnt);
    T* const cos_buf  = scratch.data() + (6 * cnt);

    T const dt_t = static_cast<T>(dt);

    // Pass 1: per site, compute β = dt·‖h‖ and stash the algebra coords.
    for (std::size_t s = 0; s < cnt; ++s) {
        std::size_t const g = base + s;
        T const h3          = p[(1 * stride) + g];
        T const h2          = p[(2 * stride) + g];
        T const h1          = p[(3 * stride) + g];
        T const hs          = (h1 * h1) + (h2 * h2) + (h3 * h3);
        T const h           = std::sqrt(hs);
        beta_buf[s]         = dt_t * h;
        h_buf[s]            = h;
        h1_buf[s]           = h1;
        h2_buf[s]           = h2;
        h3_buf[s]           = h3;
    }

    // Pass 2: vectorised sincos of β (float or double per T).
    reticolo::math::sincos_batch(sin_buf, cos_buf, beta_buf, cnt);

    // Pass 3: build V = (c, γ·h₃, γ·h₂, γ·h₁, −γ·h₂, γ·h₁, c, −γ·h₃) and
    // write U ← V · U. Branchless γ guard: at h ≈ 0 the algebra coords are
    // also ≈ 0, so any finite γ leaves V ≈ I to first order; the only worry
    // is division by zero, fixed by adding a tiny ε to ‖h‖.
    T const k_eps_h = std::numeric_limits<T>::min();
    for (std::size_t s = 0; s < cnt; ++s) {
        std::size_t const g = base + s;
        T const h           = h_buf[s];
        T const c           = cos_buf[s];
        T const sb          = sin_buf[s];
        T const gamma       = sb / (h + k_eps_h);
        T u_s[8];
        for (std::size_t k = 0; k < 8; ++k) {
            u_s[k] = u[(k * stride) + g];
        }
        T const v_s[8] = {c,
                          gamma * h3_buf[s],
                          gamma * h2_buf[s],
                          gamma * h1_buf[s],
                          -(gamma * h2_buf[s]),
                          gamma * h1_buf[s],
                          c,
                          -(gamma * h3_buf[s])};
        T o_s[8];
        mul_2x2(o_s, v_s, u_s);
        for (std::size_t k = 0; k < 8; ++k) {
            u[(k * stride) + g] = o_s[k];
        }
    }
}

// U ← exp(dt·P)·U over a full direction slab of `n` links (serial). The
// parallel drift partition lives in the integrator op layer (integ_ops.hpp),
// which calls `expi_lmul_range` per thread-chunk; this stays pure math.
template <class T>
[[gnu::always_inline]] inline void
expi_lmul_slab(T* u, T const* p, double dt, std::size_t n) noexcept {
    expi_lmul_range(u, p, dt, n, 0, n);
}

[[gnu::always_inline]] inline void project_slab(double* u, std::size_t n) noexcept {
    for (std::size_t s = 0; s < n; ++s) {
        double m[8];
        for (std::size_t k = 0; k < 8; ++k) {
            m[k] = u[(k * n) + s];
        }
        project_su2(m);
        for (std::size_t k = 0; k < 8; ++k) {
            u[(k * n) + s] = m[k];
        }
    }
}

// Sample P from the anti-hermitian-traceless Gaussian ensemble. Algebra
// coordinates h_a (a = 1,2,3) drawn i.i.d. ~ N(0, 1/√2) so that
//  Q(P) ∝ exp(−‖h‖²) = exp(−K(P))    with    K = (1/2)·Tr(P†P) = ‖h‖²
// matches the kinetic part of H used by the Metropolis accept (HMC detailed
// balance). Variance 1/2 is the SU(N) analog of the scalar N(0, 1) sampling
// against K = (1/2)·p²: in both cases σ² = 1 / (∂²K/∂coord²).
template <class T, class Rng>
[[gnu::always_inline]] inline void sample_algebra_slab(T* p, Rng& rng, std::size_t n) noexcept {
    constexpr double k_inv_sqrt2 = std::numbers::sqrt2 / 2.0;
    // Pre-fill 3n independent N(0, 1/√2) draws into a thread-local buffer,
    // then a single stride-1 scatter pass into the storage layout. Splits
    // the random-draw (rng-state-bound) from the scatter (memory-bound,
    // auto-vectorisable) — both phases run cleanly without interleaving.
    // Draws are double (the RNG stream is double); the scatter narrows to T.
    thread_local std::vector<double> h_buf;
    if (h_buf.size() < 3 * n) {
        h_buf.resize(3 * n);
    }
    rng.normal_fill(h_buf.data(), 3 * n);
    double const* const h1_arr = h_buf.data();
    double const* const h2_arr = h_buf.data() + n;
    double const* const h3_arr = h_buf.data() + (2 * n);
    for (std::size_t s = 0; s < n; ++s) {
        T const h1     = static_cast<T>(h1_arr[s] * k_inv_sqrt2);
        T const h2     = static_cast<T>(h2_arr[s] * k_inv_sqrt2);
        T const h3     = static_cast<T>(h3_arr[s] * k_inv_sqrt2);
        p[(0 * n) + s] = T{0};
        p[(1 * n) + s] = h3;
        p[(2 * n) + s] = h2;
        p[(3 * n) + s] = h1;
        p[(4 * n) + s] = -h2;
        p[(5 * n) + s] = h1;
        p[(6 * n) + s] = T{0};
        p[(7 * n) + s] = -h3;
    }
}

// Parallel counter-based momentum sampler over links [base, base+cnt) of one
// direction `mu` (component stride `stride`). Each site's 3 algebra coords come
// from Philox keyed by (key, mu, site) — a pure function of the site index, so
// the draw worksplits, is identical for any thread count, and is bit-exact on
// resume (the caller draws `key` once per trajectory from its RNG). The 1/√2
// basis scaling and 8-slot packing are identical to sample_algebra_slab; only
// the entropy source changes. Opt-in SU(2) replacement for the serial FastRng
// fill — unifies the CPU sampler with the counter-based device path.
template <class T>
inline void sample_algebra_philox_range(T* p,
                                        std::uint64_t key,
                                        std::uint64_t mu,
                                        std::size_t stride,
                                        std::size_t base,
                                        std::size_t cnt) noexcept {
    constexpr double k_inv_sqrt2 = std::numbers::sqrt2 / 2.0;
    std::size_t const end        = base + cnt;
    for (std::size_t s = base; s < end; ++s) {
        double g0 = 0.0;
        double g1 = 0.0;
        double g2 = 0.0;
        double g3 = 0.0;
        reticolo::philox_normal2(key, mu, (s * 2) + 0, g0, g1);
        reticolo::philox_normal2(key, mu, (s * 2) + 1, g2, g3);
        T const h1          = static_cast<T>(g0 * k_inv_sqrt2);
        T const h2          = static_cast<T>(g1 * k_inv_sqrt2);
        T const h3          = static_cast<T>(g2 * k_inv_sqrt2);
        p[(0 * stride) + s] = T{0};
        p[(1 * stride) + s] = h3;
        p[(2 * stride) + s] = h2;
        p[(3 * stride) + s] = h1;
        p[(4 * stride) + s] = -h2;
        p[(5 * stride) + s] = h1;
        p[(6 * stride) + s] = T{0};
        p[(7 * stride) + s] = -h3;
    }
}

// Raw Σ_{s ∈ [base, base+cnt)} (h₁² + h₂² + h₃²)  over the algebra coords of
// P. Pure per-range reduction worker — the cross-range fold is applied by the
// caller. Blocked: T-precision lane accumulators over k_b sites folded into the
// double total once per block, so a fixed k_b-block partition (see
// reduce_blocks) is thread-invariant.
template <class T>
[[gnu::always_inline]] inline double
kinetic_range(T const* p, std::size_t stride, std::size_t base, std::size_t cnt) noexcept {
    constexpr std::size_t k_b = 8;
    std::size_t const n_full  = (cnt / k_b) * k_b;
    std::size_t const tail_lo = base + n_full;
    std::size_t const end     = base + cnt;
    double k                  = 0.0;
    for (std::size_t s0 = base; s0 < tail_lo; s0 += k_b) {
        T const* h3_row = p + (1 * stride) + s0;
        T const* h2_row = p + (2 * stride) + s0;
        T const* h1_row = p + (3 * stride) + s0;
        T acc[k_b];
        for (std::size_t b = 0; b < k_b; ++b) {
            acc[b] = (h1_row[b] * h1_row[b]) + (h2_row[b] * h2_row[b]) + (h3_row[b] * h3_row[b]);
        }
        double blk = 0.0;
        for (std::size_t b = 0; b < k_b; ++b) {
            blk += static_cast<double>(acc[b]);
        }
        k += blk;
    }
    for (std::size_t s = tail_lo; s < end; ++s) {
        T const h3 = p[(1 * stride) + s];
        T const h2 = p[(2 * stride) + s];
        T const h1 = p[(3 * stride) + s];
        T const sq = (h1 * h1) + (h2 * h2) + (h3 * h3);
        k += static_cast<double>(sq);
    }
    return k;
}

// K_per_link = ||h||² where (h_1, h_2, h_3) are the algebra coords of P.
// Returns the total kinetic energy summed over n links — accumulated in double
// regardless of the field precision T (the reduction-returns-double invariant).
template <class T>
[[gnu::always_inline]] inline double kinetic_slab(T const* p, std::size_t n) noexcept {
    return kinetic_range(p, n, 0, n);
}

}  // namespace reticolo::math::su2
