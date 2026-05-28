#pragma once

#include <reticolo/action/detail/gauge_group/base.hpp>
#include <reticolo/core/indexing.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/math/su3_ops.hpp>

#include <cstddef>
#include <string_view>
#include <vector>

namespace reticolo::gauge_group {

// SU(3) gauge group model. Storage: full 3×3 complex matrix, 18 reals per
// link element (see math::su3 for the layout). N = 3, so the Wilson
// prefactor is β/3 and the per-link constant in s_full is 3·n_plaq.
//
// Same surface as `SU2`: GaugeGroup concept members + per-plaquette
// `plaq_re_tr` + HMC slab hooks (sample/kinetic/expi_lmul/project) + a
// link-centric `compute_force` that walks each (x, μ) and assembles the
// full staple sum + TA[U·V] in one pass.

struct SU3 {
    using scalar_t                                 = double;
    static constexpr std::size_t n_real_components = 18;
    static constexpr std::size_t n_color           = 3;
    static constexpr std::string_view name         = "SU3";

    // Re Tr (AB · DC†) where AB = U_μ(s)·U_ν(s+μ), DC = U_ν(s)·U_μ(s+ν).
    // Same identity as SU(2): Re Tr (X · Y†) = sum_{ij} [Re X_{ij}·Re Y_{ij}
    // + Im X_{ij}·Im Y_{ij}] = 18-component inner product of X and Y.
    template <class T>
    [[gnu::always_inline]] static inline double plaq_re_tr(T const* mb,
                                                           T const* nb,
                                                           std::size_t s,
                                                           std::size_t s_pmu,
                                                           std::size_t s_pnu,
                                                           std::size_t stride) noexcept {
        double a_mat[18];
        double b_mat[18];
        double c_mat[18];
        double d_mat[18];
        for (std::size_t k = 0; k < 18; ++k) {
            std::size_t const off = k * stride;
            a_mat[k]              = static_cast<double>(mb[off + s]);
            b_mat[k]              = static_cast<double>(nb[off + s_pmu]);
            c_mat[k]              = static_cast<double>(mb[off + s_pnu]);
            d_mat[k]              = static_cast<double>(nb[off + s]);
        }
        double ab[18];
        double dc[18];
        math::su3::mul_3x3(ab, a_mat, b_mat);
        math::su3::mul_3x3(dc, d_mat, c_mat);
        double re_tr = 0.0;
        for (std::size_t k = 0; k < 18; ++k) {
            re_tr += ab[k] * dc[k];
        }
        return re_tr;
    }

    // -------- HMC slab hooks --------------------------------------------------
    template <class T, class Rng>
    [[gnu::always_inline]] static inline void
    sample_algebra_slab(T* p_blk, Rng& rng, std::size_t n) noexcept {
        math::su3::sample_algebra_slab(p_blk, rng, n);
    }

    template <class T>
    [[gnu::always_inline]] static inline double kinetic_slab(T const* p_blk,
                                                             std::size_t n) noexcept {
        return math::su3::kinetic_slab(p_blk, n);
    }

    template <class T>
    [[gnu::always_inline]] static inline void
    expi_lmul_slab(T* u_blk, T const* p_blk, double dt, std::size_t n) noexcept {
        math::su3::expi_lmul_slab(u_blk, p_blk, dt, n);
    }

    // -------- link-centric Wilson force --------------------------------------
    // Same shape as SU2::compute_force, just with 18-real matrices.
    //   V_μ(x) = sum_{ν ≠ μ} [ fwd_μν(x) + bwd_μν(x) ]
    //   F_μ(x) = −(β/N) · TA[U_μ(x) · V_μ(x)]
    //
    // Body shared between the plain force (out = F) and the fused kick
    // (mom += k_dt · F) via a templated `Fused` non-type parameter.
private:
    // ---------------- Portable batched fast path ---------------------------
    //
    // Batched version of compute_force_impl_ that processes K_BATCH sites at
    // a time. All the matrix math is written as nested loops whose innermost
    // loop walks `for b in 0..K_BATCH` over stride-1 packed-per-site buffers.
    // The compiler auto-vectorises the b-loop into whatever SIMD width the
    // target supports (NEON 2-wide, AVX2 4-wide, AVX-512 8-wide).
    //
    // Index lookups (s_pmu, s_pnu, ...) stay scalar (the next/prev tables are
    // gathers we don't try to vectorise). Loads from `u_*_blk[k*ns + idx]` are
    // therefore done as a B-element scalar gather into an AoSoA scratch — the
    // compiler often turns these into SIMD gathers on targets that support
    // them, but even when it doesn't the matmul that follows is fully SIMD.
    //
    // No interleaved (re, im) lanes — they live in separate `..._re[9][B]`
    // and `..._im[9][B]` scratch buffers, so each `cr += ar*br - ai*bi` /
    // `ci += ar*bi + ai*br` becomes pure stride-1 vector arithmetic with no
    // cross-lane permutation. That's the structural fix that frees the
    // auto-vectoriser to pack `b`.
    static constexpr std::size_t k_batch = k_gauge_batch;

    template <bool Fused, class T>
    [[gnu::always_inline]] static inline void compute_force_batched_(
        MatrixLinkLattice<SU3, T> const& u, MatrixLinkLattice<SU3, T>& out, double scale) noexcept {
        std::size_t const d  = u.ndims();
        std::size_t const ns = u.nsites();
        Indexing const& idx  = u.indexing_ref();

        std::size_t const n_batches = ns / k_batch;
        std::size_t const tail_base = n_batches * k_batch;
        // Force math runs at the field precision T: float packs k_batch sites
        // into 4-wide lanes (8-wide on AVX2), double into 2-wide. No widen.
        T const scl = static_cast<T>(scale);

        for (std::size_t mu = 0; mu < d; ++mu) {
            T const* const u_mu_blk = u.mu_block_data(mu);
            T* const out_mu_blk     = out.mu_block_data(mu);

            for (std::size_t bi = 0; bi < n_batches; ++bi) {
                std::size_t const s_base = bi * k_batch;

                // ---- Neighbour indices for this batch ------------------
                std::size_t s_pmu[k_batch];
                for (std::size_t b = 0; b < k_batch; ++b) {
                    s_pmu[b] = idx.next(Site{s_base + b}, mu).value();
                }

                // ---- Staple accumulator V_re/V_im[9][B] ----------------
                T v_re[9][k_batch];
                T v_im[9][k_batch];
                for (std::size_t k = 0; k < 9; ++k) {
                    for (std::size_t b = 0; b < k_batch; ++b) {
                        v_re[k][b] = T{0};
                        v_im[k][b] = T{0};
                    }
                }

                for (std::size_t nu = 0; nu < d; ++nu) {
                    if (nu == mu) {
                        continue;
                    }
                    T const* const u_nu_blk = u.mu_block_data(nu);

                    std::size_t s_pnu[k_batch];
                    std::size_t s_mnu[k_batch];
                    std::size_t s_pmu_mnu[k_batch];
                    for (std::size_t b = 0; b < k_batch; ++b) {
                        s_pnu[b]     = idx.next(Site{s_base + b}, nu).value();
                        s_mnu[b]     = idx.prev(Site{s_base + b}, nu).value();
                        s_pmu_mnu[b] = idx.prev(Site{s_pmu[b]}, nu).value();
                    }

                    // ------------- Forward staple ---------------------------
                    // t1 = U_ν(s+μ̂) · U_μ(s+ν̂)† ,  t2 = t1 · U_ν(s)† ,  v += t2
                    T a_re[9][k_batch];
                    T a_im[9][k_batch];
                    T b_re[9][k_batch];
                    T b_im[9][k_batch];
                    T c_re[9][k_batch];
                    T c_im[9][k_batch];

                    for (std::size_t k = 0; k < 9; ++k) {
                        std::size_t const off_re = (2 * k) * ns;
                        std::size_t const off_im = ((2 * k) + 1) * ns;
                        for (std::size_t b = 0; b < k_batch; ++b) {
                            a_re[k][b] = u_nu_blk[off_re + s_pmu[b]];
                            a_im[k][b] = u_nu_blk[off_im + s_pmu[b]];
                            b_re[k][b] = u_mu_blk[off_re + s_pnu[b]];
                            b_im[k][b] = u_mu_blk[off_im + s_pnu[b]];
                            c_re[k][b] = u_nu_blk[off_re + s_base + b];
                            c_im[k][b] = u_nu_blk[off_im + s_base + b];
                        }
                    }

                    // t1 = a · b†   →   t1_{ij} = sum_k a_{ik} · conj(b_{jk})
                    // (ar+iai)(br-ibi) = (ar·br + ai·bi) + i·(ai·br - ar·bi)
                    T t1_re[9][k_batch];
                    T t1_im[9][k_batch];
                    for (std::size_t i = 0; i < 3; ++i) {
                        for (std::size_t j = 0; j < 3; ++j) {
                            T cr[k_batch];
                            T ci[k_batch];
                            for (std::size_t b = 0; b < k_batch; ++b) {
                                cr[b] = T{0};
                                ci[b] = T{0};
                            }
                            for (std::size_t k = 0; k < 3; ++k) {
                                std::size_t const ka = (3 * i) + k;
                                std::size_t const kb = (3 * j) + k;
                                for (std::size_t b = 0; b < k_batch; ++b) {
                                    cr[b] +=
                                        (a_re[ka][b] * b_re[kb][b]) + (a_im[ka][b] * b_im[kb][b]);
                                    ci[b] +=
                                        (a_im[ka][b] * b_re[kb][b]) - (a_re[ka][b] * b_im[kb][b]);
                                }
                            }
                            std::size_t const out_k = (3 * i) + j;
                            for (std::size_t b = 0; b < k_batch; ++b) {
                                t1_re[out_k][b] = cr[b];
                                t1_im[out_k][b] = ci[b];
                            }
                        }
                    }

                    // v += t1 · c†   →   (t1_{ik}) · conj(c_{jk}) — same shape.
                    for (std::size_t i = 0; i < 3; ++i) {
                        for (std::size_t j = 0; j < 3; ++j) {
                            T cr[k_batch];
                            T ci[k_batch];
                            for (std::size_t b = 0; b < k_batch; ++b) {
                                cr[b] = T{0};
                                ci[b] = T{0};
                            }
                            for (std::size_t k = 0; k < 3; ++k) {
                                std::size_t const ka = (3 * i) + k;
                                std::size_t const kb = (3 * j) + k;
                                for (std::size_t b = 0; b < k_batch; ++b) {
                                    cr[b] +=
                                        (t1_re[ka][b] * c_re[kb][b]) + (t1_im[ka][b] * c_im[kb][b]);
                                    ci[b] +=
                                        (t1_im[ka][b] * c_re[kb][b]) - (t1_re[ka][b] * c_im[kb][b]);
                                }
                            }
                            std::size_t const out_k = (3 * i) + j;
                            for (std::size_t b = 0; b < k_batch; ++b) {
                                v_re[out_k][b] += cr[b];
                                v_im[out_k][b] += ci[b];
                            }
                        }
                    }

                    // ------------- Backward staple --------------------------
                    // t1 = U_μ(s-ν̂)† · U_ν(s-ν̂) ,  t2 = U_ν(s+μ̂-ν̂)† · t1
                    // ar+iai)† · (br+ibi) form for t1 (a† · b style):
                    //   (ar-iai)(br+ibi) = (ar·br + ai·bi) + i·(ar·bi - ai·br)
                    for (std::size_t k = 0; k < 9; ++k) {
                        std::size_t const off_re = (2 * k) * ns;
                        std::size_t const off_im = ((2 * k) + 1) * ns;
                        for (std::size_t b = 0; b < k_batch; ++b) {
                            a_re[k][b] = u_nu_blk[off_re + s_pmu_mnu[b]];
                            a_im[k][b] = u_nu_blk[off_im + s_pmu_mnu[b]];
                            b_re[k][b] = u_mu_blk[off_re + s_mnu[b]];
                            b_im[k][b] = u_mu_blk[off_im + s_mnu[b]];
                            c_re[k][b] = u_nu_blk[off_re + s_mnu[b]];
                            c_im[k][b] = u_nu_blk[off_im + s_mnu[b]];
                        }
                    }
                    // t1 = b† · c   →   t1_{ij} = sum_k conj(b_{ki}) · c_{kj}
                    for (std::size_t i = 0; i < 3; ++i) {
                        for (std::size_t j = 0; j < 3; ++j) {
                            T cr[k_batch];
                            T ci[k_batch];
                            for (std::size_t b = 0; b < k_batch; ++b) {
                                cr[b] = T{0};
                                ci[b] = T{0};
                            }
                            for (std::size_t k = 0; k < 3; ++k) {
                                std::size_t const ka = (3 * k) + i;
                                std::size_t const kb = (3 * k) + j;
                                for (std::size_t b = 0; b < k_batch; ++b) {
                                    cr[b] +=
                                        (b_re[ka][b] * c_re[kb][b]) + (b_im[ka][b] * c_im[kb][b]);
                                    ci[b] +=
                                        (b_re[ka][b] * c_im[kb][b]) - (b_im[ka][b] * c_re[kb][b]);
                                }
                            }
                            std::size_t const out_k = (3 * i) + j;
                            for (std::size_t b = 0; b < k_batch; ++b) {
                                t1_re[out_k][b] = cr[b];
                                t1_im[out_k][b] = ci[b];
                            }
                        }
                    }
                    // v += a† · t1
                    for (std::size_t i = 0; i < 3; ++i) {
                        for (std::size_t j = 0; j < 3; ++j) {
                            T cr[k_batch];
                            T ci[k_batch];
                            for (std::size_t b = 0; b < k_batch; ++b) {
                                cr[b] = T{0};
                                ci[b] = T{0};
                            }
                            for (std::size_t k = 0; k < 3; ++k) {
                                std::size_t const ka = (3 * k) + i;
                                std::size_t const kb = (3 * k) + j;
                                for (std::size_t b = 0; b < k_batch; ++b) {
                                    cr[b] +=
                                        (a_re[ka][b] * t1_re[kb][b]) + (a_im[ka][b] * t1_im[kb][b]);
                                    ci[b] +=
                                        (a_re[ka][b] * t1_im[kb][b]) - (a_im[ka][b] * t1_re[kb][b]);
                                }
                            }
                            std::size_t const out_k = (3 * i) + j;
                            for (std::size_t b = 0; b < k_batch; ++b) {
                                v_re[out_k][b] += cr[b];
                                v_im[out_k][b] += ci[b];
                            }
                        }
                    }
                }

                // ------------ Final: U_μ(s) · V → TA → scatter ------------
                T u_re[9][k_batch];
                T u_im[9][k_batch];
                for (std::size_t k = 0; k < 9; ++k) {
                    std::size_t const off_re = (2 * k) * ns;
                    std::size_t const off_im = ((2 * k) + 1) * ns;
                    for (std::size_t b = 0; b < k_batch; ++b) {
                        u_re[k][b] = u_mu_blk[off_re + s_base + b];
                        u_im[k][b] = u_mu_blk[off_im + s_base + b];
                    }
                }
                // uv = U · V (plain mul):
                //   c_{ij} = sum_k U_{ik} · V_{kj}
                //   (ar+iai)(br+ibi) = (ar·br - ai·bi) + i·(ar·bi + ai·br)
                T uv_re[9][k_batch];
                T uv_im[9][k_batch];
                for (std::size_t i = 0; i < 3; ++i) {
                    for (std::size_t j = 0; j < 3; ++j) {
                        T cr[k_batch];
                        T ci[k_batch];
                        for (std::size_t b = 0; b < k_batch; ++b) {
                            cr[b] = T{0};
                            ci[b] = T{0};
                        }
                        for (std::size_t k = 0; k < 3; ++k) {
                            std::size_t const ka = (3 * i) + k;
                            std::size_t const kb = (3 * k) + j;
                            for (std::size_t b = 0; b < k_batch; ++b) {
                                cr[b] += (u_re[ka][b] * v_re[kb][b]) - (u_im[ka][b] * v_im[kb][b]);
                                ci[b] += (u_re[ka][b] * v_im[kb][b]) + (u_im[ka][b] * v_re[kb][b]);
                            }
                        }
                        std::size_t const out_k = (3 * i) + j;
                        for (std::size_t b = 0; b < k_batch; ++b) {
                            uv_re[out_k][b] = cr[b];
                            uv_im[out_k][b] = ci[b];
                        }
                    }
                }
                // TA(M): diagonal = i·(Im_{ii} − Tr/3); off-diag (i<j):
                //   re_ij_out = 0.5·(Re_{ij} − Re_{ji})
                //   im_ij_out = 0.5·(Im_{ij} + Im_{ji})
                //   re_ji_out = −re_ij_out ;  im_ji_out = im_ij_out
                //   diag re = 0; diag im = Im_{ii} − T/3 where T = sum Im_{ii}.
                T ta_re[9][k_batch];
                T ta_im[9][k_batch];
                for (std::size_t b = 0; b < k_batch; ++b) {
                    T const t_over_3 = (uv_im[0][b] + uv_im[4][b] + uv_im[8][b]) / T{3};
                    ta_re[0][b]      = T{0};
                    ta_im[0][b]      = uv_im[0][b] - t_over_3;
                    ta_re[4][b]      = T{0};
                    ta_im[4][b]      = uv_im[4][b] - t_over_3;
                    ta_re[8][b]      = T{0};
                    ta_im[8][b]      = uv_im[8][b] - t_over_3;
                    // (0,1) vs (1,0)
                    T const re01 = T{0.5} * (uv_re[1][b] - uv_re[3][b]);
                    T const im01 = T{0.5} * (uv_im[1][b] + uv_im[3][b]);
                    ta_re[1][b]  = re01;
                    ta_im[1][b]  = im01;
                    ta_re[3][b]  = -re01;
                    ta_im[3][b]  = im01;
                    // (0,2) vs (2,0)
                    T const re02 = T{0.5} * (uv_re[2][b] - uv_re[6][b]);
                    T const im02 = T{0.5} * (uv_im[2][b] + uv_im[6][b]);
                    ta_re[2][b]  = re02;
                    ta_im[2][b]  = im02;
                    ta_re[6][b]  = -re02;
                    ta_im[6][b]  = im02;
                    // (1,2) vs (2,1)
                    T const re12 = T{0.5} * (uv_re[5][b] - uv_re[7][b]);
                    T const im12 = T{0.5} * (uv_im[5][b] + uv_im[7][b]);
                    ta_re[5][b]  = re12;
                    ta_im[5][b]  = im12;
                    ta_re[7][b]  = -re12;
                    ta_im[7][b]  = im12;
                }
                // Scatter scale·ta into out_mu_blk.
                for (std::size_t k = 0; k < 9; ++k) {
                    std::size_t const off_re = (2 * k) * ns;
                    std::size_t const off_im = ((2 * k) + 1) * ns;
                    if constexpr (Fused) {
                        for (std::size_t b = 0; b < k_batch; ++b) {
                            out_mu_blk[off_re + s_base + b] += scl * ta_re[k][b];
                            out_mu_blk[off_im + s_base + b] += scl * ta_im[k][b];
                        }
                    } else {
                        for (std::size_t b = 0; b < k_batch; ++b) {
                            out_mu_blk[off_re + s_base + b] = scl * ta_re[k][b];
                            out_mu_blk[off_im + s_base + b] = scl * ta_im[k][b];
                        }
                    }
                }
            }
        }

        // Scalar tail for the last ns % k_batch sites (rare — only when ns
        // isn't a multiple of 8, e.g. odd lattice volumes).
        if (tail_base < ns) {
            compute_force_impl_tail_<Fused>(u, out, scale, tail_base);
        }
    }

    // Scalar fallback covering [tail_base, ns). Same math as
    // compute_force_impl_ but restricted to a flat s range — reused by the
    // batched path's remainder loop.
    template <bool Fused, class T>
    [[gnu::always_inline]] static inline void
    compute_force_impl_tail_(MatrixLinkLattice<SU3, T> const& u,
                             MatrixLinkLattice<SU3, T>& out,
                             double scale,
                             std::size_t s_start) noexcept {
        std::size_t const d  = u.ndims();
        std::size_t const ns = u.nsites();
        Indexing const& idx  = u.indexing_ref();

        auto load_link = [ns](double* dst, auto const* blk, std::size_t s) noexcept {
            for (std::size_t k = 0; k < 18; ++k) {
                dst[k] = static_cast<double>(blk[(k * ns) + s]);
            }
        };

        for (std::size_t mu = 0; mu < d; ++mu) {
            T const* const u_mu_blk = u.mu_block_data(mu);
            T* const out_mu_blk     = out.mu_block_data(mu);
            for (std::size_t s = s_start; s < ns; ++s) {
                double v[18] = {};
                Site const x{s};
                std::size_t const s_pmu = idx.next(x, mu).value();
                for (std::size_t nu = 0; nu < d; ++nu) {
                    if (nu == mu) {
                        continue;
                    }
                    T const* const u_nu_blk     = u.mu_block_data(nu);
                    std::size_t const s_pnu     = idx.next(x, nu).value();
                    std::size_t const s_mnu     = idx.prev(x, nu).value();
                    std::size_t const s_pmu_mnu = idx.prev(Site{s_pmu}, nu).value();
                    {
                        double a[18];
                        double b[18];
                        double c[18];
                        double t1[18];
                        double t2[18];
                        load_link(a, u_nu_blk, s_pmu);
                        load_link(b, u_mu_blk, s_pnu);
                        load_link(c, u_nu_blk, s);
                        math::su3::mul_adj_3x3(t1, a, b);
                        math::su3::mul_adj_3x3(t2, t1, c);
                        for (std::size_t k = 0; k < 18; ++k) {
                            v[k] += t2[k];
                        }
                    }
                    {
                        double a[18];
                        double b[18];
                        double c[18];
                        double t1[18];
                        double t2[18];
                        load_link(a, u_nu_blk, s_pmu_mnu);
                        load_link(b, u_mu_blk, s_mnu);
                        load_link(c, u_nu_blk, s_mnu);
                        math::su3::adj_mul_3x3(t1, b, c);
                        math::su3::adj_mul_3x3(t2, a, t1);
                        for (std::size_t k = 0; k < 18; ++k) {
                            v[k] += t2[k];
                        }
                    }
                }
                double u_s[18];
                double uv[18];
                double ta[18];
                load_link(u_s, u_mu_blk, s);
                math::su3::mul_3x3(uv, u_s, v);
                math::su3::traceless_antiherm_3x3(ta, uv);
                if constexpr (Fused) {
                    for (std::size_t k = 0; k < 18; ++k) {
                        out_mu_blk[(k * ns) + s] += static_cast<T>(scale * ta[k]);
                    }
                } else {
                    for (std::size_t k = 0; k < 18; ++k) {
                        out_mu_blk[(k * ns) + s] = static_cast<T>(scale * ta[k]);
                    }
                }
            }
        }
    }

public:
    // Precision-generic: the batched kernel is T-native (float → 4-/8-wide,
    // double → 2-wide) and folds the ns % k_batch remainder into the per-site
    // tail, so both precisions take the same path. The batched kernel writes
    // each (μ, s) exactly once, so no force zero-init is needed (Fused=false).
    template <class T>
    static void compute_force(MatrixLinkLattice<SU3, T> const& u,
                              MatrixLinkLattice<SU3, T>& force,
                              double beta_over_n) noexcept {
        compute_force_batched_<false>(u, force, -beta_over_n);
    }

    template <class T>
    static void compute_force_and_kick(MatrixLinkLattice<SU3, T> const& u,
                                       MatrixLinkLattice<SU3, T>& mom,
                                       double beta_over_n,
                                       double k_dt) noexcept {
        compute_force_batched_<true>(u, mom, -k_dt * beta_over_n);
    }
};

static_assert(GaugeGroup<SU3>);

}  // namespace reticolo::gauge_group
