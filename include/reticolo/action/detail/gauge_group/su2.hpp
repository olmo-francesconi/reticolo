#pragma once

#include <reticolo/action/detail/gauge_group/base.hpp>
#include <reticolo/core/indexing.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/math/su2_ops.hpp>

#include <cstddef>
#include <string_view>
#include <type_traits>

namespace reticolo::gauge_group {

// =============================================================================
//  SU(2) gauge group model. Storage: full 2×2 complex matrix, 8 reals per
//  link element (see math::su2 for the component layout). N = 2, so the
//  Wilson prefactor is β/2 and the per-link constant in s_full is 2·n_plaq.
//
//  This M4 commit ships the GaugeGroup-concept-mandatory members plus the
//  `plaq_re_tr` primitive that `Wilson<G>::s_full` and `s_local` need —
//  enough to *compute* the Wilson action on an SU(2) configuration without
//  yet having an HMC force kernel. `plaq_force_accum` (and therefore
//  `Wilson<SU2>::compute_force`) arrives in M5 once the staple-scatter
//  pattern is exercised end-to-end.
// =============================================================================

struct SU2 {
    using scalar_t                                 = double;
    static constexpr std::size_t n_real_components = 8;
    static constexpr std::size_t n_color           = 2;
    static constexpr std::string_view name         = "SU2";

    // Re Tr (U_mu(s) · U_nu(s+pmu) · U_mu(s+pnu)† · U_nu(s)†)
    // = Re Tr (AB · DC†) where AB = U_mu(s)·U_nu(s+pmu) and DC = U_nu(s)·U_mu(s+pnu).
    // Using Re Tr (X · Y†) = sum_{ij} [Re X_{ij}·Re Y_{ij} + Im X_{ij}·Im Y_{ij}]
    // — the inner product of the two matrices viewed as 8-real vectors.
    template <class T>
    [[gnu::always_inline]] static inline double plaq_re_tr(T const* mb,
                                                           T const* nb,
                                                           std::size_t s,
                                                           std::size_t s_pmu,
                                                           std::size_t s_pnu,
                                                           std::size_t stride) noexcept {
        double a_mat[8];
        double b_mat[8];
        double c_mat[8];
        double d_mat[8];
        for (std::size_t k = 0; k < 8; ++k) {
            std::size_t const off = k * stride;
            a_mat[k]              = static_cast<double>(mb[off + s]);
            b_mat[k]              = static_cast<double>(nb[off + s_pmu]);
            c_mat[k]              = static_cast<double>(mb[off + s_pnu]);
            d_mat[k]              = static_cast<double>(nb[off + s]);
        }
        double ab[8];
        double dc[8];
        math::su2::mul_2x2(ab, a_mat, b_mat);
        math::su2::mul_2x2(dc, d_mat, c_mat);
        double re_tr = 0.0;
        for (std::size_t k = 0; k < 8; ++k) {
            re_tr += ab[k] * dc[k];
        }
        return re_tr;
    }

    // -------- HMC slab hooks (raw-pointer thin wrappers over math::su2) ------
    // These keep `Hmc` field-agnostic: HMC loops directions and calls
    // `G::sample_algebra_slab` / `kinetic_slab` / `expi_lmul_slab` per
    // direction. For non-`double` T the wrappers simply don't instantiate —
    // matrix-link SU(2) is a `double`-only field type in this implementation.
    template <class Rng>
    [[gnu::always_inline]] static inline void
    sample_algebra_slab(double* p_blk, Rng& rng, std::size_t n) noexcept {
        math::su2::sample_algebra_slab(p_blk, rng, n);
    }

    [[gnu::always_inline]] static inline double kinetic_slab(double const* p_blk,
                                                             std::size_t n) noexcept {
        return math::su2::kinetic_slab(p_blk, n);
    }

    [[gnu::always_inline]] static inline void
    expi_lmul_slab(double* u_blk, double const* p_blk, double dt, std::size_t n) noexcept {
        math::su2::expi_lmul_slab(u_blk, p_blk, dt, n);
    }

    // -------- link-centric Wilson force --------------------------------------
    //
    //   V_μ(x) = sum_{ν ≠ μ} [ fwd_μν(x) + bwd_μν(x) ]
    //   fwd_μν(x) = U_ν(x+μ) · U_μ(x+ν)† · U_ν(x)†
    //   bwd_μν(x) = U_ν(x+μ−ν)† · U_μ(x−ν)† · U_ν(x−ν)
    //   F_μ(x) = −(β/N) · TA[U_μ(x) · V_μ(x)]
    //
    // Per link: 2(d−1) staples · 2 cmm each + 1 cmm + TA + scalar mul.
    //
    // The body is shared between the plain force (out = F) and the fused
    // kick (mom += k_dt · F) via a templated `Fused` non-type parameter so
    // the integrator can skip materialising a force buffer entirely.
private:
    template <bool Fused, class T>
    [[gnu::always_inline]] static inline void compute_force_impl_(
        MatrixLinkLattice<SU2, T> const& u, MatrixLinkLattice<SU2, T>& out, double scale) noexcept {
        std::size_t const d  = u.ndims();
        std::size_t const ns = u.nsites();
        Indexing const& idx  = u.indexing_ref();

        auto load_link = [ns](double* dst, T const* blk, std::size_t s) noexcept {
            for (std::size_t k = 0; k < 8; ++k) {
                dst[k] = static_cast<double>(blk[(k * ns) + s]);
            }
        };

        for (std::size_t mu = 0; mu < d; ++mu) {
            T const* const u_mu_blk = u.mu_block_data(mu);
            T* const out_mu_blk     = out.mu_block_data(mu);
            for (std::size_t s = 0; s < ns; ++s) {
                double v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
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

                    // Forward staple: U_ν(s+μ) · U_μ(s+ν)† · U_ν(s)†
                    {
                        double a[8];
                        double b[8];
                        double c[8];
                        double t1[8];
                        double t2[8];
                        load_link(a, u_nu_blk, s_pmu);
                        load_link(b, u_mu_blk, s_pnu);
                        load_link(c, u_nu_blk, s);
                        math::su2::mul_adj_2x2(t1, a, b);
                        math::su2::mul_adj_2x2(t2, t1, c);
                        for (std::size_t k = 0; k < 8; ++k) {
                            v[k] += t2[k];
                        }
                    }
                    // Backward staple: U_ν(s+μ−ν)† · U_μ(s−ν)† · U_ν(s−ν)
                    {
                        double a[8];
                        double b[8];
                        double c[8];
                        double t1[8];
                        double t2[8];
                        load_link(a, u_nu_blk, s_pmu_mnu);
                        load_link(b, u_mu_blk, s_mnu);
                        load_link(c, u_nu_blk, s_mnu);
                        math::su2::adj_mul_2x2(t1, b, c);
                        math::su2::adj_mul_2x2(t2, a, t1);
                        for (std::size_t k = 0; k < 8; ++k) {
                            v[k] += t2[k];
                        }
                    }
                }
                double u_s[8];
                double uv[8];
                double ta[8];
                load_link(u_s, u_mu_blk, s);
                math::su2::mul_2x2(uv, u_s, v);
                math::su2::traceless_antiherm_2x2(ta, uv);
                if constexpr (Fused) {
                    for (std::size_t k = 0; k < 8; ++k) {
                        out_mu_blk[(k * ns) + s] += static_cast<T>(scale * ta[k]);
                    }
                } else {
                    for (std::size_t k = 0; k < 8; ++k) {
                        out_mu_blk[(k * ns) + s] = static_cast<T>(scale * ta[k]);
                    }
                }
            }
        }
    }

    // ---------------- Portable batched fast path (k_batch sites) -----------
    //
    // Mirrors the SU(3) batched kernel: split Re/Im into separate AoSoA
    // slabs, run all matmul / TA / scatter loops with an innermost
    // `for b in 0..k_batch` over stride-1 packed data. The compiler
    // auto-vectorises the b-loop on any target SIMD width.
    static constexpr std::size_t k_batch = 8;

    template <bool Fused, class T>
    [[gnu::always_inline]] static inline void compute_force_batched_(
        MatrixLinkLattice<SU2, T> const& u, MatrixLinkLattice<SU2, T>& out, double scale) noexcept {
        std::size_t const d  = u.ndims();
        std::size_t const ns = u.nsites();
        Indexing const& idx  = u.indexing_ref();

        std::size_t const n_batches = ns / k_batch;
        std::size_t const tail_base = n_batches * k_batch;

        for (std::size_t mu = 0; mu < d; ++mu) {
            T const* const u_mu_blk = u.mu_block_data(mu);
            T* const out_mu_blk     = out.mu_block_data(mu);

            for (std::size_t bi = 0; bi < n_batches; ++bi) {
                std::size_t const s_base = bi * k_batch;

                std::size_t s_pmu[k_batch];
                for (std::size_t b = 0; b < k_batch; ++b) {
                    s_pmu[b] = idx.next(Site{s_base + b}, mu).value();
                }

                // V accumulator (4 complex entries → 4 Re + 4 Im).
                double v_re[4][k_batch];
                double v_im[4][k_batch];
                for (std::size_t k = 0; k < 4; ++k) {
                    for (std::size_t b = 0; b < k_batch; ++b) {
                        v_re[k][b] = 0.0;
                        v_im[k][b] = 0.0;
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

                    // -------- Forward staple ------------------------------
                    double a_re[4][k_batch];
                    double a_im[4][k_batch];
                    double b_re[4][k_batch];
                    double b_im[4][k_batch];
                    double c_re[4][k_batch];
                    double c_im[4][k_batch];
                    for (std::size_t k = 0; k < 4; ++k) {
                        std::size_t const off_re = (2 * k) * ns;
                        std::size_t const off_im = ((2 * k) + 1) * ns;
                        for (std::size_t b = 0; b < k_batch; ++b) {
                            a_re[k][b] = static_cast<double>(u_nu_blk[off_re + s_pmu[b]]);
                            a_im[k][b] = static_cast<double>(u_nu_blk[off_im + s_pmu[b]]);
                            b_re[k][b] = static_cast<double>(u_mu_blk[off_re + s_pnu[b]]);
                            b_im[k][b] = static_cast<double>(u_mu_blk[off_im + s_pnu[b]]);
                            c_re[k][b] = static_cast<double>(u_nu_blk[off_re + s_base + b]);
                            c_im[k][b] = static_cast<double>(u_nu_blk[off_im + s_base + b]);
                        }
                    }

                    // t1 = a · b†   →   t1_{ij} = sum_k a_{ik} · conj(b_{jk})
                    // 2×2: k ∈ {0, 1}; matrix entry at slot 2*i + j.
                    double t1_re[4][k_batch];
                    double t1_im[4][k_batch];
                    for (std::size_t i = 0; i < 2; ++i) {
                        for (std::size_t j = 0; j < 2; ++j) {
                            double cr[k_batch];
                            double ci[k_batch];
                            for (std::size_t b = 0; b < k_batch; ++b) {
                                cr[b] = 0.0;
                                ci[b] = 0.0;
                            }
                            for (std::size_t k = 0; k < 2; ++k) {
                                std::size_t const ka = (2 * i) + k;
                                std::size_t const kb = (2 * j) + k;
                                for (std::size_t b = 0; b < k_batch; ++b) {
                                    cr[b] +=
                                        (a_re[ka][b] * b_re[kb][b]) + (a_im[ka][b] * b_im[kb][b]);
                                    ci[b] +=
                                        (a_im[ka][b] * b_re[kb][b]) - (a_re[ka][b] * b_im[kb][b]);
                                }
                            }
                            std::size_t const out_k = (2 * i) + j;
                            for (std::size_t b = 0; b < k_batch; ++b) {
                                t1_re[out_k][b] = cr[b];
                                t1_im[out_k][b] = ci[b];
                            }
                        }
                    }

                    // v += t1 · c†
                    for (std::size_t i = 0; i < 2; ++i) {
                        for (std::size_t j = 0; j < 2; ++j) {
                            double cr[k_batch];
                            double ci[k_batch];
                            for (std::size_t b = 0; b < k_batch; ++b) {
                                cr[b] = 0.0;
                                ci[b] = 0.0;
                            }
                            for (std::size_t k = 0; k < 2; ++k) {
                                std::size_t const ka = (2 * i) + k;
                                std::size_t const kb = (2 * j) + k;
                                for (std::size_t b = 0; b < k_batch; ++b) {
                                    cr[b] +=
                                        (t1_re[ka][b] * c_re[kb][b]) + (t1_im[ka][b] * c_im[kb][b]);
                                    ci[b] +=
                                        (t1_im[ka][b] * c_re[kb][b]) - (t1_re[ka][b] * c_im[kb][b]);
                                }
                            }
                            std::size_t const out_k = (2 * i) + j;
                            for (std::size_t b = 0; b < k_batch; ++b) {
                                v_re[out_k][b] += cr[b];
                                v_im[out_k][b] += ci[b];
                            }
                        }
                    }

                    // -------- Backward staple -----------------------------
                    for (std::size_t k = 0; k < 4; ++k) {
                        std::size_t const off_re = (2 * k) * ns;
                        std::size_t const off_im = ((2 * k) + 1) * ns;
                        for (std::size_t b = 0; b < k_batch; ++b) {
                            a_re[k][b] = static_cast<double>(u_nu_blk[off_re + s_pmu_mnu[b]]);
                            a_im[k][b] = static_cast<double>(u_nu_blk[off_im + s_pmu_mnu[b]]);
                            b_re[k][b] = static_cast<double>(u_mu_blk[off_re + s_mnu[b]]);
                            b_im[k][b] = static_cast<double>(u_mu_blk[off_im + s_mnu[b]]);
                            c_re[k][b] = static_cast<double>(u_nu_blk[off_re + s_mnu[b]]);
                            c_im[k][b] = static_cast<double>(u_nu_blk[off_im + s_mnu[b]]);
                        }
                    }
                    // t1 = b† · c
                    for (std::size_t i = 0; i < 2; ++i) {
                        for (std::size_t j = 0; j < 2; ++j) {
                            double cr[k_batch];
                            double ci[k_batch];
                            for (std::size_t b = 0; b < k_batch; ++b) {
                                cr[b] = 0.0;
                                ci[b] = 0.0;
                            }
                            for (std::size_t k = 0; k < 2; ++k) {
                                std::size_t const ka = (2 * k) + i;
                                std::size_t const kb = (2 * k) + j;
                                for (std::size_t b = 0; b < k_batch; ++b) {
                                    cr[b] +=
                                        (b_re[ka][b] * c_re[kb][b]) + (b_im[ka][b] * c_im[kb][b]);
                                    ci[b] +=
                                        (b_re[ka][b] * c_im[kb][b]) - (b_im[ka][b] * c_re[kb][b]);
                                }
                            }
                            std::size_t const out_k = (2 * i) + j;
                            for (std::size_t b = 0; b < k_batch; ++b) {
                                t1_re[out_k][b] = cr[b];
                                t1_im[out_k][b] = ci[b];
                            }
                        }
                    }
                    // v += a† · t1
                    for (std::size_t i = 0; i < 2; ++i) {
                        for (std::size_t j = 0; j < 2; ++j) {
                            double cr[k_batch];
                            double ci[k_batch];
                            for (std::size_t b = 0; b < k_batch; ++b) {
                                cr[b] = 0.0;
                                ci[b] = 0.0;
                            }
                            for (std::size_t k = 0; k < 2; ++k) {
                                std::size_t const ka = (2 * k) + i;
                                std::size_t const kb = (2 * k) + j;
                                for (std::size_t b = 0; b < k_batch; ++b) {
                                    cr[b] +=
                                        (a_re[ka][b] * t1_re[kb][b]) + (a_im[ka][b] * t1_im[kb][b]);
                                    ci[b] +=
                                        (a_re[ka][b] * t1_im[kb][b]) - (a_im[ka][b] * t1_re[kb][b]);
                                }
                            }
                            std::size_t const out_k = (2 * i) + j;
                            for (std::size_t b = 0; b < k_batch; ++b) {
                                v_re[out_k][b] += cr[b];
                                v_im[out_k][b] += ci[b];
                            }
                        }
                    }
                }

                // ------------ Final: U_μ(s) · V → TA → scatter ------------
                double u_re[4][k_batch];
                double u_im[4][k_batch];
                for (std::size_t k = 0; k < 4; ++k) {
                    std::size_t const off_re = (2 * k) * ns;
                    std::size_t const off_im = ((2 * k) + 1) * ns;
                    for (std::size_t b = 0; b < k_batch; ++b) {
                        u_re[k][b] = static_cast<double>(u_mu_blk[off_re + s_base + b]);
                        u_im[k][b] = static_cast<double>(u_mu_blk[off_im + s_base + b]);
                    }
                }
                double uv_re[4][k_batch];
                double uv_im[4][k_batch];
                for (std::size_t i = 0; i < 2; ++i) {
                    for (std::size_t j = 0; j < 2; ++j) {
                        double cr[k_batch];
                        double ci[k_batch];
                        for (std::size_t b = 0; b < k_batch; ++b) {
                            cr[b] = 0.0;
                            ci[b] = 0.0;
                        }
                        for (std::size_t k = 0; k < 2; ++k) {
                            std::size_t const ka = (2 * i) + k;
                            std::size_t const kb = (2 * k) + j;
                            for (std::size_t b = 0; b < k_batch; ++b) {
                                cr[b] += (u_re[ka][b] * v_re[kb][b]) - (u_im[ka][b] * v_im[kb][b]);
                                ci[b] += (u_re[ka][b] * v_im[kb][b]) + (u_im[ka][b] * v_re[kb][b]);
                            }
                        }
                        std::size_t const out_k = (2 * i) + j;
                        for (std::size_t b = 0; b < k_batch; ++b) {
                            uv_re[out_k][b] = cr[b];
                            uv_im[out_k][b] = ci[b];
                        }
                    }
                }
                // SU(2) TA: 8-slot layout (k=0..7) is
                //   (Re_{00} Im_{00} Re_{01} Im_{01} Re_{10} Im_{10} Re_{11} Im_{11})
                // i.e. slot 2k+0,1 = entry index k ∈ {(0,0),(0,1),(1,0),(1,1)}.
                // TA = (M − M†)/2 − Tr/2 · I, diagonal becomes pure imag,
                // off-diag is the anti-hermitian completion of (M_{01}, M_{10}).
                double ta_re[4][k_batch];
                double ta_im[4][k_batch];
                for (std::size_t b = 0; b < k_batch; ++b) {
                    double const diag_im = 0.5 * (uv_im[0][b] - uv_im[3][b]);
                    ta_re[0][b]          = 0.0;
                    ta_im[0][b]          = diag_im;
                    double const re01    = 0.5 * (uv_re[1][b] - uv_re[2][b]);
                    double const im01    = 0.5 * (uv_im[1][b] + uv_im[2][b]);
                    ta_re[1][b]          = re01;
                    ta_im[1][b]          = im01;
                    ta_re[2][b]          = -re01;
                    ta_im[2][b]          = im01;
                    ta_re[3][b]          = 0.0;
                    ta_im[3][b]          = -diag_im;
                }
                for (std::size_t k = 0; k < 4; ++k) {
                    std::size_t const off_re = (2 * k) * ns;
                    std::size_t const off_im = ((2 * k) + 1) * ns;
                    if constexpr (Fused) {
                        for (std::size_t b = 0; b < k_batch; ++b) {
                            out_mu_blk[off_re + s_base + b] += static_cast<T>(scale * ta_re[k][b]);
                            out_mu_blk[off_im + s_base + b] += static_cast<T>(scale * ta_im[k][b]);
                        }
                    } else {
                        for (std::size_t b = 0; b < k_batch; ++b) {
                            out_mu_blk[off_re + s_base + b] = static_cast<T>(scale * ta_re[k][b]);
                            out_mu_blk[off_im + s_base + b] = static_cast<T>(scale * ta_im[k][b]);
                        }
                    }
                }
            }
        }

        if (tail_base < ns) {
            compute_force_impl_tail_<Fused>(u, out, scale, tail_base);
        }
    }

    template <bool Fused, class T>
    [[gnu::always_inline]] static inline void
    compute_force_impl_tail_(MatrixLinkLattice<SU2, T> const& u,
                             MatrixLinkLattice<SU2, T>& out,
                             double scale,
                             std::size_t s_start) noexcept {
        std::size_t const d  = u.ndims();
        std::size_t const ns = u.nsites();
        Indexing const& idx  = u.indexing_ref();

        auto load_link = [ns](double* dst, auto const* blk, std::size_t s) noexcept {
            for (std::size_t k = 0; k < 8; ++k) {
                dst[k] = static_cast<double>(blk[(k * ns) + s]);
            }
        };

        for (std::size_t mu = 0; mu < d; ++mu) {
            T const* const u_mu_blk = u.mu_block_data(mu);
            T* const out_mu_blk     = out.mu_block_data(mu);
            for (std::size_t s = s_start; s < ns; ++s) {
                double v[8] = {0, 0, 0, 0, 0, 0, 0, 0};
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
                        double a[8];
                        double b[8];
                        double c[8];
                        double t1[8];
                        double t2[8];
                        load_link(a, u_nu_blk, s_pmu);
                        load_link(b, u_mu_blk, s_pnu);
                        load_link(c, u_nu_blk, s);
                        math::su2::mul_adj_2x2(t1, a, b);
                        math::su2::mul_adj_2x2(t2, t1, c);
                        for (std::size_t k = 0; k < 8; ++k) {
                            v[k] += t2[k];
                        }
                    }
                    {
                        double a[8];
                        double b[8];
                        double c[8];
                        double t1[8];
                        double t2[8];
                        load_link(a, u_nu_blk, s_pmu_mnu);
                        load_link(b, u_mu_blk, s_mnu);
                        load_link(c, u_nu_blk, s_mnu);
                        math::su2::adj_mul_2x2(t1, b, c);
                        math::su2::adj_mul_2x2(t2, a, t1);
                        for (std::size_t k = 0; k < 8; ++k) {
                            v[k] += t2[k];
                        }
                    }
                }
                double u_s[8];
                double uv[8];
                double ta[8];
                load_link(u_s, u_mu_blk, s);
                math::su2::mul_2x2(uv, u_s, v);
                math::su2::traceless_antiherm_2x2(ta, uv);
                if constexpr (Fused) {
                    for (std::size_t k = 0; k < 8; ++k) {
                        out_mu_blk[(k * ns) + s] += static_cast<T>(scale * ta[k]);
                    }
                } else {
                    for (std::size_t k = 0; k < 8; ++k) {
                        out_mu_blk[(k * ns) + s] = static_cast<T>(scale * ta[k]);
                    }
                }
            }
        }
    }

public:
    template <class T>
    static void compute_force(MatrixLinkLattice<SU2, T> const& u,
                              MatrixLinkLattice<SU2, T>& force,
                              double beta_over_n) noexcept {
        if constexpr (std::is_same_v<T, double>) {
            compute_force_batched_<false>(u, force, -beta_over_n);
        } else {
            compute_force_impl_tail_<false>(u, force, -beta_over_n, 0);
        }
    }

    template <class T>
    static void compute_force_and_kick(MatrixLinkLattice<SU2, T> const& u,
                                       MatrixLinkLattice<SU2, T>& mom,
                                       double beta_over_n,
                                       double k_dt) noexcept {
        if constexpr (std::is_same_v<T, double>) {
            compute_force_batched_<true>(u, mom, -k_dt * beta_over_n);
        } else {
            compute_force_impl_tail_<true>(u, mom, -k_dt * beta_over_n, 0);
        }
    }
};

static_assert(GaugeGroup<SU2>);

}  // namespace reticolo::gauge_group
