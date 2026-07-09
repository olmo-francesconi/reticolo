#pragma once

#include <reticolo/action/gauge/formula/wilson_kernels.hpp>
#include <reticolo/action/sweep/plane.hpp>
#include <reticolo/core/indexing.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/parallel.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/math/group/su2.hpp>
#include <reticolo/math/su2_ops.hpp>

#include <cstddef>

// Wilson plaquette kernels for SU(2) — the action-specific physics (Re Tr U_p,
// the Σ Re Tr plane fast-path, the link-centric staple force / fused kick) for
// the Wilson action on a MatrixLinkLattice<math::group::SU2, T>. Kept out of the SU2 group
// model, which holds only the core group operations.

namespace reticolo::action::formula {

template <>
struct wilson_kernels<math::group::SU2> {
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

    // Σ Re Tr U_p over the sites [base, base+cnt) of one (μ, ν) plane. Pure per-
    // range worker — no threading. The gauge base parallelises by calling this
    // over a fixed block partition (field_reduce); `base` is k_batch-
    // aligned so the batched groupings match the whole-plane sweep exactly.
    // Same shape as SU3::s_full_plane_range, [4][B] slabs.
    //
    // When L0 isn't a multiple of the batch, every row-crossing batch loses
    // load contiguity and the 2×2 math (~50 FLOPs/site) is too light to
    // amortise the gathers — measured ~25% slower than the plane-walking
    // scalar path on 3D L=12. Misaligned shapes take the per-site route
    // instead; SU(3)'s heavier math wins batched either way, so only SU(2)
    // guards.
    template <class T>
    static double s_full_plane_range(MatrixLinkLattice<math::group::SU2, T> const& u,
                                     std::size_t mu,
                                     std::size_t nu,
                                     std::size_t base,
                                     std::size_t cnt) noexcept {
        constexpr std::size_t k_batch = math::group::k_gauge_batch<T>;
        std::size_t const ns          = u.link_span();  // padded component stride
        Indexing const& idx           = u.indexing_ref();
        T const* const u_mu_blk       = u.mu_block_data(mu);
        T const* const u_nu_blk       = u.mu_block_data(nu);

        if (u.shape()[0] % k_batch != 0) {
            double total          = 0.0;
            std::size_t const end = base + cnt;
            for (std::size_t s = base; s < end; ++s) {
                std::size_t const s_pmu = idx.next(Site{s}, mu).value();
                std::size_t const s_pnu = idx.next(Site{s}, nu).value();
                total += plaq_re_tr(u_mu_blk, u_nu_blk, s, s_pmu, s_pnu, ns);
            }
            return total;
        }

        std::size_t const n_full  = (cnt / k_batch) * k_batch;  // full-batch part
        std::size_t const tail_lo = base + n_full;
        std::size_t const end     = base + cnt;
        double total              = 0.0;

        for (std::size_t s_base = base; s_base < tail_lo; s_base += k_batch) {
            std::size_t s_pmu[k_batch];
            std::size_t s_pnu[k_batch];
            for (std::size_t b = 0; b < k_batch; ++b) {
                s_pmu[b] = idx.next(Site{s_base + b}, mu).value();
                s_pnu[b] = idx.next(Site{s_base + b}, nu).value();
            }
            // A = U_μ(s), B = U_ν(s+μ̂), C = U_μ(s+ν̂), D = U_ν(s).
            T a_re[4][k_batch];
            T a_im[4][k_batch];
            T b_re[4][k_batch];
            T b_im[4][k_batch];
            T c_re[4][k_batch];
            T c_im[4][k_batch];
            T d_re[4][k_batch];
            T d_im[4][k_batch];
            math::group::load_links_batched(a_re, a_im, u_mu_blk, ns, s_base);
            math::group::load_links_batched(b_re, b_im, u_nu_blk, ns, s_pmu);
            math::group::load_links_batched(c_re, c_im, u_mu_blk, ns, s_pnu);
            math::group::load_links_batched(d_re, d_im, u_nu_blk, ns, s_base);
            T ab_re[4][k_batch];
            T ab_im[4][k_batch];
            T dc_re[4][k_batch];
            T dc_im[4][k_batch];
            math::su2::mul_2x2_batched<false>(ab_re, ab_im, a_re, a_im, b_re, b_im);
            math::su2::mul_2x2_batched<false>(dc_re, dc_im, d_re, d_im, c_re, c_im);
            // Re Tr (AB · DC†) = Σ_k [Re·Re + Im·Im] — 8-real inner product.
            T acc[k_batch];
            for (std::size_t b = 0; b < k_batch; ++b) {
                acc[b] = T{0};
            }
            for (std::size_t k = 0; k < 4; ++k) {
                for (std::size_t b = 0; b < k_batch; ++b) {
                    acc[b] += (ab_re[k][b] * dc_re[k][b]) + (ab_im[k][b] * dc_im[k][b]);
                }
            }
            for (std::size_t b = 0; b < k_batch; ++b) {
                total += static_cast<double>(acc[b]);
            }
        }
        for (std::size_t s = tail_lo; s < end; ++s) {
            std::size_t const s_pmu = idx.next(Site{s}, mu).value();
            std::size_t const s_pnu = idx.next(Site{s}, nu).value();
            total += plaq_re_tr(u_mu_blk, u_nu_blk, s, s_pmu, s_pnu, ns);
        }
        return total;
    }

    // Whole-plane Σ Re Tr U_p (serial). The `Wilson<G>::s_full` present/absent
    // probe binds to this name; the parallel path calls `s_full_plane_range`
    // per block via field_reduce.
    template <class T>
    static double s_full_plane_re_tr_sum(MatrixLinkLattice<math::group::SU2, T> const& u,
                                         std::size_t mu,
                                         std::size_t nu) noexcept {
        return s_full_plane_range<T>(u, mu, nu, 0, u.nsites());
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
public:
    // ---------------- Portable batched fast path (k_batch sites) -----------
    //
    // Mirrors the SU(3) batched kernel: split Re/Im into separate AoSoA
    // slabs, run all matmul / TA / scatter loops with an innermost
    // `for b in 0..k_batch` over stride-1 packed data. The compiler
    // auto-vectorises the b-loop on any target SIMD width.
    //
    // Pure per-range staple force/kick worker over the links [base, base+cnt).
    // No threading — the gauge base parallelises via field_visit, whose
    // k_batch-aligned `base` keeps every non-final chunk on the batched path so
    // the result is bit-identical to the whole-field sweep. Each (μ, s) is written
    // exactly once, so chunks are write-disjoint.
    template <bool Fused, class T>
    [[gnu::always_inline]] static inline void
    compute_force_range(MatrixLinkLattice<math::group::SU2, T> const& u,
                        MatrixLinkLattice<math::group::SU2, T>& out,
                        double scale,
                        std::size_t base,
                        std::size_t cnt) noexcept {
        constexpr std::size_t k_batch = math::group::k_gauge_batch<T>;
        std::size_t const d           = u.ndims();
        std::size_t const ns          = u.link_span();  // padded component stride
        Indexing const& idx           = u.indexing_ref();

        std::size_t const n_full  = (cnt / k_batch) * k_batch;  // full-batch part
        std::size_t const tail_lo = base + n_full;
        std::size_t const end     = base + cnt;
        // Force math runs at the field precision T: float links pack k_batch
        // sites into 4-wide lanes, double into 2-wide. No widen-to-double.
        T const scl = static_cast<T>(scale);

        for (std::size_t mu = 0; mu < d; ++mu) {
            T const* const u_mu_blk = u.mu_block_data(mu);
            T* const out_mu_blk     = out.mu_block_data(mu);

            for (std::size_t s_base = base; s_base < tail_lo; s_base += k_batch) {
                std::size_t s_pmu[k_batch];
                for (std::size_t b = 0; b < k_batch; ++b) {
                    s_pmu[b] = idx.next(Site{s_base + b}, mu).value();
                }

                // V accumulator (4 complex entries → 4 Re + 4 Im).
                T v_re[4][k_batch];
                T v_im[4][k_batch];
                for (std::size_t k = 0; k < 4; ++k) {
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

                    // -------- Forward staple ------------------------------
                    T a_re[4][k_batch];
                    T a_im[4][k_batch];
                    T b_re[4][k_batch];
                    T b_im[4][k_batch];
                    T c_re[4][k_batch];
                    T c_im[4][k_batch];
                    math::group::load_links_batched(a_re, a_im, u_nu_blk, ns, s_pmu);
                    math::group::load_links_batched(b_re, b_im, u_mu_blk, ns, s_pnu);
                    math::group::load_links_batched(c_re, c_im, u_nu_blk, ns, s_base);

                    // t1 = a · b† ,  v += t1 · c†
                    T t1_re[4][k_batch];
                    T t1_im[4][k_batch];
                    math::su2::mul_adj_2x2_batched<false>(t1_re, t1_im, a_re, a_im, b_re, b_im);
                    math::su2::mul_adj_2x2_batched<true>(v_re, v_im, t1_re, t1_im, c_re, c_im);

                    // -------- Backward staple -----------------------------
                    math::group::load_links_batched(a_re, a_im, u_nu_blk, ns, s_pmu_mnu);
                    math::group::load_links_batched(b_re, b_im, u_mu_blk, ns, s_mnu);
                    math::group::load_links_batched(c_re, c_im, u_nu_blk, ns, s_mnu);
                    // t1 = b† · c ,  v += a† · t1
                    math::su2::adj_mul_2x2_batched<false>(t1_re, t1_im, b_re, b_im, c_re, c_im);
                    math::su2::adj_mul_2x2_batched<true>(v_re, v_im, a_re, a_im, t1_re, t1_im);
                }

                // ------------ Final: U_μ(s) · V → TA → scatter ------------
                T u_re[4][k_batch];
                T u_im[4][k_batch];
                math::group::load_links_batched(u_re, u_im, u_mu_blk, ns, s_base);
                // uv = U · V, then TA into the algebra.
                T uv_re[4][k_batch];
                T uv_im[4][k_batch];
                math::su2::mul_2x2_batched<false>(uv_re, uv_im, u_re, u_im, v_re, v_im);
                T ta_re[4][k_batch];
                T ta_im[4][k_batch];
                math::su2::traceless_antiherm_2x2_batched(ta_re, ta_im, uv_re, uv_im);
                for (std::size_t k = 0; k < 4; ++k) {
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

        // Scalar tail for the chunk's [tail_lo, end) remainder (nonzero only on
        // the final chunk, carrying the global ns % k_batch sites).
        if (tail_lo < end) {
            compute_force_impl_tail_<Fused>(u, out, scale, tail_lo, end);
        }
    }

private:
    // Scalar fallback covering [s_start, s_end). Same math as the batched
    // kernel but per-site. Cold path (only when ns % k_batch ≠ 0) — kept out
    // of line so its straight-line FP ops don't double the I-cache footprint
    // of every force instantiation.
    template <bool Fused, class T>
    [[gnu::noinline]] static void
    compute_force_impl_tail_(MatrixLinkLattice<math::group::SU2, T> const& u,
                             MatrixLinkLattice<math::group::SU2, T>& out,
                             double scale,
                             std::size_t s_start,
                             std::size_t s_end) noexcept {
        std::size_t const d  = u.ndims();
        std::size_t const ns = u.link_span();  // padded component stride
        Indexing const& idx  = u.indexing_ref();

        auto load_link = [ns](double* dst, auto const* blk, std::size_t s) noexcept {
            for (std::size_t k = 0; k < 8; ++k) {
                dst[k] = static_cast<double>(blk[(k * ns) + s]);
            }
        };

        for (std::size_t mu = 0; mu < d; ++mu) {
            T const* const u_mu_blk = u.mu_block_data(mu);
            T* const out_mu_blk     = out.mu_block_data(mu);
            for (std::size_t s = s_start; s < s_end; ++s) {
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
    // The batched kernel is precision-generic (T-native: float → 4-wide,
    // double → 2-wide) and handles the ns % k_batch remainder via the per-site
    // tail, so both precisions take the same path. Write-disjoint per (μ, s), so
    // `field_visit` worksplits it over k_batch-aligned chunks bit-identically.
    template <class T>
    static void compute_force(MatrixLinkLattice<math::group::SU2, T> const& u,
                              MatrixLinkLattice<math::group::SU2, T>& force,
                              double beta_over_n) noexcept {
        reticolo::exec::field_visit(
            u, math::group::k_gauge_batch<T>, [&](std::size_t base, std::size_t cnt) {
                compute_force_range<false>(u, force, -beta_over_n, base, cnt);
            });
    }

    template <class T>
    static void compute_force_and_kick(MatrixLinkLattice<math::group::SU2, T> const& u,
                                       MatrixLinkLattice<math::group::SU2, T>& mom,
                                       double beta_over_n,
                                       double k_dt) noexcept {
        reticolo::exec::field_visit(
            u, math::group::k_gauge_batch<T>, [&](std::size_t base, std::size_t cnt) {
                compute_force_range<true>(u, mom, -k_dt * beta_over_n, base, cnt);
            });
    }
};

}  // namespace reticolo::action::formula
