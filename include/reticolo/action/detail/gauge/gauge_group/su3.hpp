#pragma once

#include <reticolo/action/detail/gauge/gauge_group/base.hpp>
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

    // Batched Σ Re Tr U_p over one (μ, ν) plane — the `Wilson<G>::s_full`
    // fast path. T-native arithmetic (float keeps its full lane count, no
    // widen-to-double per component); per-batch partials are folded into the
    // double accumulator so the volume-sized sum keeps the double invariant
    // and the b-loops stay free of a loop-carried double chain.
    template <class T>
    static double s_full_plane_re_tr_sum(MatrixLinkLattice<SU3, T> const& u,
                                         std::size_t mu,
                                         std::size_t nu) noexcept {
        constexpr std::size_t k_batch = k_gauge_batch<T>;
        std::size_t const ns          = u.nsites();
        Indexing const& idx           = u.indexing_ref();
        T const* const u_mu_blk       = u.mu_block_data(mu);
        T const* const u_nu_blk       = u.mu_block_data(nu);

        std::size_t const n_batches = ns / k_batch;
        std::size_t const tail_base = n_batches * k_batch;
        double total                = 0.0;

        for (std::size_t bi = 0; bi < n_batches; ++bi) {
            std::size_t const s_base = bi * k_batch;
            std::size_t s_pmu[k_batch];
            std::size_t s_pnu[k_batch];
            for (std::size_t b = 0; b < k_batch; ++b) {
                s_pmu[b] = idx.next(Site{s_base + b}, mu).value();
                s_pnu[b] = idx.next(Site{s_base + b}, nu).value();
            }
            // A = U_μ(s), B = U_ν(s+μ̂), C = U_μ(s+ν̂), D = U_ν(s).
            T a_re[9][k_batch];
            T a_im[9][k_batch];
            T b_re[9][k_batch];
            T b_im[9][k_batch];
            T c_re[9][k_batch];
            T c_im[9][k_batch];
            T d_re[9][k_batch];
            T d_im[9][k_batch];
            load_links_batched(a_re, a_im, u_mu_blk, ns, s_base);
            load_links_batched(b_re, b_im, u_nu_blk, ns, s_pmu);
            load_links_batched(c_re, c_im, u_mu_blk, ns, s_pnu);
            load_links_batched(d_re, d_im, u_nu_blk, ns, s_base);
            T ab_re[9][k_batch];
            T ab_im[9][k_batch];
            T dc_re[9][k_batch];
            T dc_im[9][k_batch];
            math::su3::mul_3x3_batched<false>(ab_re, ab_im, a_re, a_im, b_re, b_im);
            math::su3::mul_3x3_batched<false>(dc_re, dc_im, d_re, d_im, c_re, c_im);
            // Re Tr (AB · DC†) = Σ_k [Re·Re + Im·Im] — 18-real inner product.
            T acc[k_batch];
            for (std::size_t b = 0; b < k_batch; ++b) {
                acc[b] = T{0};
            }
            for (std::size_t k = 0; k < 9; ++k) {
                for (std::size_t b = 0; b < k_batch; ++b) {
                    acc[b] += (ab_re[k][b] * dc_re[k][b]) + (ab_im[k][b] * dc_im[k][b]);
                }
            }
            for (std::size_t b = 0; b < k_batch; ++b) {
                total += static_cast<double>(acc[b]);
            }
        }
        for (std::size_t s = tail_base; s < ns; ++s) {
            std::size_t const s_pmu = idx.next(Site{s}, mu).value();
            std::size_t const s_pnu = idx.next(Site{s}, nu).value();
            total += plaq_re_tr(u_mu_blk, u_nu_blk, s, s_pmu, s_pnu, ns);
        }
        return total;
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
    template <bool Fused, class T>
    [[gnu::always_inline]] static inline void compute_force_batched_(
        MatrixLinkLattice<SU3, T> const& u, MatrixLinkLattice<SU3, T>& out, double scale) noexcept {
        constexpr std::size_t k_batch = k_gauge_batch<T>;
        std::size_t const d           = u.ndims();
        std::size_t const ns          = u.nsites();
        Indexing const& idx           = u.indexing_ref();

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
                    load_links_batched(a_re, a_im, u_nu_blk, ns, s_pmu);
                    load_links_batched(b_re, b_im, u_mu_blk, ns, s_pnu);
                    load_links_batched(c_re, c_im, u_nu_blk, ns, s_base);

                    // t1 = a · b† ,  v += t1 · c†
                    T t1_re[9][k_batch];
                    T t1_im[9][k_batch];
                    math::su3::mul_adj_3x3_batched<false>(t1_re, t1_im, a_re, a_im, b_re, b_im);
                    math::su3::mul_adj_3x3_batched<true>(v_re, v_im, t1_re, t1_im, c_re, c_im);

                    // ------------- Backward staple --------------------------
                    // t1 = U_μ(s-ν̂)† · U_ν(s-ν̂) ,  t2 = U_ν(s+μ̂-ν̂)† · t1
                    load_links_batched(a_re, a_im, u_nu_blk, ns, s_pmu_mnu);
                    load_links_batched(b_re, b_im, u_mu_blk, ns, s_mnu);
                    load_links_batched(c_re, c_im, u_nu_blk, ns, s_mnu);
                    // t1 = b† · c ,  v += a† · t1
                    math::su3::adj_mul_3x3_batched<false>(t1_re, t1_im, b_re, b_im, c_re, c_im);
                    math::su3::adj_mul_3x3_batched<true>(v_re, v_im, a_re, a_im, t1_re, t1_im);
                }

                // ------------ Final: U_μ(s) · V → TA → scatter ------------
                T u_re[9][k_batch];
                T u_im[9][k_batch];
                load_links_batched(u_re, u_im, u_mu_blk, ns, s_base);
                // uv = U · V, then TA into the algebra.
                T uv_re[9][k_batch];
                T uv_im[9][k_batch];
                math::su3::mul_3x3_batched<false>(uv_re, uv_im, u_re, u_im, v_re, v_im);
                T ta_re[9][k_batch];
                T ta_im[9][k_batch];
                math::su3::traceless_antiherm_3x3_batched(ta_re, ta_im, uv_re, uv_im);
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

    // Scalar fallback covering [tail_base, ns). Same math as the batched
    // kernel but per-site. Cold path (only when ns % k_batch ≠ 0) — kept out
    // of line so its ~900 straight-line FP ops don't double the I-cache
    // footprint of every force instantiation.
    template <bool Fused, class T>
    [[gnu::noinline]] static void compute_force_impl_tail_(MatrixLinkLattice<SU3, T> const& u,
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
