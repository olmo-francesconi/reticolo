#pragma once

#include <reticolo/core/exec/parallel.hpp>
#include <reticolo/core/rng/philox.hpp>
#include <reticolo/math/group/formula/su2_formula.hpp>
#include <reticolo/math/vec_libm.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numbers>
#include <vector>

namespace reticolo::math::su2 {

// NOTE: the per-link formulas (cmul_acc*, mul_2x2/mul_adj_2x2/adj_mul_2x2,
// traceless_antiherm_2x2, exp_su2) are single-sourced in
// math/group/formula/su2_formula.hpp, RETICOLO_HD so the CUDA device path
// (cuda::SU2Device, cuda/gauge/su2_device.cuh) forwards to them directly
// instead of re-implementing them — mechanically gated by
// tests/cuda/check_no_gauge_kernels.cmake, not just test_cuda_su2.
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

// ---------- batched AoSoA 2×2 complex products -------------------------------
// Operate on `[4][B]` re/im slabs: entry (i,j) of site b lives at
// `re[(2*i)+j][b]` / `im[(2*i)+j][b]`. The innermost b-loop is stride-1
// packed data, so the compiler vectorises it at the target SIMD width.
// `Acc` selects `out +=` over `out =`. Out must not alias the inputs.

// out = a · b
template <std::size_t B, class T>
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
                out_re[out_k][b] = cr[b];
                out_im[out_k][b] = ci[b];
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

// ---------- closed-form group projection --------------------------------------

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
    reticolo::exec::thread_scratch(scratch, 7 * cnt);
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
// parallel drift partition lives in the field-op layer (core/exec/field_ops.hpp),
// which calls `expi_lmul_range` per thread-chunk; this stays pure math.
template <class T>
[[gnu::always_inline]] inline void
expi_lmul_slab(T* u, T const* p, double dt, std::size_t n) noexcept {
    expi_lmul_range(u, p, dt, n, 0, n);
}

// Padded form: `count` links at component stride `stride` (u and p share it).
template <class T>
[[gnu::always_inline]] inline void
expi_lmul_slab(T* u, T const* p, double dt, std::size_t stride, std::size_t count) noexcept {
    expi_lmul_range(u, p, dt, stride, 0, count);
}

// Sample P from the anti-hermitian-traceless Gaussian ensemble. Algebra
// coordinates h_a (a = 1,2,3) drawn i.i.d. ~ N(0, 1/√2) so that
//  Q(P) ∝ exp(−‖h‖²) = exp(−K(P))    with    K = (1/2)·Tr(P†P) = ‖h‖²
// matches the kinetic part of H used by the Metropolis accept (HMC detailed
// balance). Variance 1/2 is the SU(N) analog of the scalar N(0, 1) sampling
// against K = (1/2)·p²: in both cases σ² = 1 / (∂²K/∂coord²).
// (stride, count) form: write `count` links' algebra with component stride
// `stride` (≥ count, padded ≥ nsites for cache-friendly gauge storage). The
// 3·count-normal draw order is a pure function of `count`, so stride == count
// is bit-identical to the packed layout (the 2-arg overload forwards to it).
template <class T, class Rng>
[[gnu::always_inline]] inline void
sample_algebra_slab(T* p, Rng& rng, std::size_t stride, std::size_t count) noexcept {
    constexpr double k_inv_sqrt2 = std::numbers::sqrt2 / 2.0;
    // Pre-fill 3·count independent N(0, 1/√2) draws into a thread-local buffer,
    // then a single stride-1 scatter pass into the storage layout. Splits
    // the random-draw (rng-state-bound) from the scatter (memory-bound,
    // auto-vectorisable) — both phases run cleanly without interleaving.
    // Draws are double (the RNG stream is double); the scatter narrows to T.
    thread_local std::vector<double> h_buf;
    double* const h = reticolo::exec::thread_scratch(h_buf, 3 * count);
    rng.normal_fill(h, 3 * count);
    double const* const h1_arr = h;
    double const* const h2_arr = h + count;
    double const* const h3_arr = h + (2 * count);
    for (std::size_t s = 0; s < count; ++s) {
        T const h1          = static_cast<T>(h1_arr[s] * k_inv_sqrt2);
        T const h2          = static_cast<T>(h2_arr[s] * k_inv_sqrt2);
        T const h3          = static_cast<T>(h3_arr[s] * k_inv_sqrt2);
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

// Packed convenience: stride == count == n.
template <class T, class Rng>
[[gnu::always_inline]] inline void sample_algebra_slab(T* p, Rng& rng, std::size_t n) noexcept {
    sample_algebra_slab(p, rng, n, n);
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

// Σ_{s ∈ [base, base+cnt)} Tr(P† P) = 2·(h₁² + h₂² + h₃²)  over the algebra
// coords of P (the factor 2 is Tr(σ_a σ_b) = 2 δ_ab, i.e. summing the 8 stored
// reals of P: {0,h₃,h₂,h₁,−h₂,h₁,0,−h₃} squared). Matches the SU(3) convention
// so the shared HMC kinetic — K = 0.5·Σ kinetic_range = ‖h‖² — is consistent
// with the variance-½ momentum sampling. Pure per-range reduction worker — the
// cross-range fold is applied by the caller. Blocked: T-precision lane
// accumulators over k_b sites folded into the double total once per block, so a
// fixed k_b-block partition (see parallel_reduce) is thread-invariant.
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
        for (auto& b : acc) {
            blk += static_cast<double>(b);
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
    return 2.0 * k;  // Tr(P†P) = 2‖h‖²
}

// K_per_link = ‖h‖² = (1/2) Tr(P† P), summed over the links — accumulated in
// double regardless of the field precision T (the reduction-returns-double
// invariant). The ½ mirrors SU(3) so kinetic_range alone stays Tr(P†P).
template <class T>
[[gnu::always_inline]] inline double kinetic_slab(T const* p, std::size_t n) noexcept {
    return 0.5 * kinetic_range(p, n, 0, n);
}

template <class T>
[[gnu::always_inline]] inline double
kinetic_slab(T const* p, std::size_t stride, std::size_t count) noexcept {
    return 0.5 * kinetic_range(p, stride, 0, count);
}

}  // namespace reticolo::math::su2
