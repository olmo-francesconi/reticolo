#pragma once

#include <reticolo/action/gauge/formula/wilson_kernels.hpp>
#include <reticolo/core/field/indexing.hpp>
#include <reticolo/core/field/matrix_link_lattice.hpp>
#include <reticolo/core/exec/parallel.hpp>
#include <reticolo/core/field/site.hpp>
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

    // Table-free row-nested plane sum: the two forward neighbours (s+μ̂, s+ν̂) are
    // computed from the lattice strides, so the k_batch index arrays stay
    // contiguous (stride-1 vector loads) and no Indexing table is read. Unifies
    // the former shape[0] % k_batch guard: that split existed because the GATHER
    // was slow on misaligned shapes — with strided contiguous loads the batched
    // path wins throughout, with the row remainder handled per site.
    template <class T>
    static double s_full_plane_range(MatrixLinkLattice<math::group::SU2, T> const& u,
                                     std::size_t mu,
                                     std::size_t nu,
                                     std::size_t base,
                                     std::size_t cnt) noexcept {
        constexpr std::size_t k_batch = math::group::k_gauge_batch<T>;
        auto const& sh                = u.shape();
        std::size_t const d           = u.ndims();
        std::size_t const span        = u.link_span();  // padded component stride
        std::size_t const l0          = sh[0];
        T const* const u_mu_blk       = u.mu_block_data(mu);
        T const* const u_nu_blk       = u.mu_block_data(nu);

        std::size_t stg[4] = {1, 0, 0, 0};
        for (std::size_t j = 1; j < d; ++j) {
            stg[j] = stg[j - 1] * sh[j - 1];
        }
        std::size_t const r0    = base / l0;
        std::size_t const nrows = cnt / l0;
        double total            = 0.0;

        for (std::size_t r = r0; r < r0 + nrows; ++r) {
            std::size_t const row = r * l0;
            std::size_t fwd[4]    = {0, 0, 0, 0};
            bool fwd_wrap[4]      = {false, false, false, false};
            std::size_t rr        = r;
            for (std::size_t j = 1; j < d; ++j) {
                std::size_t const lj = sh[j];
                std::size_t const cj = rr % lj;
                rr /= lj;
                fwd[j]      = (cj + 1 < lj) ? stg[j] : (lj - 1) * stg[j];
                fwd_wrap[j] = (cj + 1 == lj);
            }
            auto nxt = [&](std::size_t s, std::size_t x, std::size_t j) -> std::size_t {
                if (j == 0) {
                    return (x + 1 == l0) ? s - (l0 - 1) : s + 1;
                }
                return fwd_wrap[j] ? s - fwd[j] : s + fwd[j];
            };

            std::size_t x = 0;
            for (; x + k_batch <= l0; x += k_batch) {
                std::size_t const s_base = row + x;
                std::size_t s_pmu[k_batch];
                std::size_t s_pnu[k_batch];
                for (std::size_t b = 0; b < k_batch; ++b) {
                    s_pmu[b] = nxt(s_base + b, x + b, mu);
                    s_pnu[b] = nxt(s_base + b, x + b, nu);
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
                math::group::load_links_batched(a_re, a_im, u_mu_blk, span, s_base);
                math::group::load_links_batched(b_re, b_im, u_nu_blk, span, s_pmu);
                math::group::load_links_batched(c_re, c_im, u_mu_blk, span, s_pnu);
                math::group::load_links_batched(d_re, d_im, u_nu_blk, span, s_base);
                T ab_re[4][k_batch];
                T ab_im[4][k_batch];
                T dc_re[4][k_batch];
                T dc_im[4][k_batch];
                math::su2::mul_2x2_batched(ab_re, ab_im, a_re, a_im, b_re, b_im);
                math::su2::mul_2x2_batched(dc_re, dc_im, d_re, d_im, c_re, c_im);
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
            for (; x < l0; ++x) {
                std::size_t const s = row + x;
                total += plaq_re_tr(u_mu_blk, u_nu_blk, s, nxt(s, x, mu), nxt(s, x, nu), span);
            }
        }
        return total;
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
    // Table-free: neighbour flat indices come from the lattice strides via a
    // row-nested sweep (odometer over the outer dims, dim 0 as the inner loop),
    // NOT the Indexing next/prev tables. A direction ≥ 1 neighbour is a per-row
    // constant offset, so the k_batch index array stays contiguous and
    // load_links_batched collapses it to a stride-1 vector load; direction 0
    // wraps per site. The staple math + scatter are byte-for-byte the former
    // table sweep, so the force is bit-identical. Dimension-generic for
    // 2 ≤ d ≤ 4 (enforced on the gauge field). The partition hands whole rows
    // (base, cnt are multiples of L0), so each (μ, s) is written once →
    // write-disjoint over the range.
    template <bool Fused, class T>
    static void compute_force_range(MatrixLinkLattice<math::group::SU2, T> const& u,
                                    MatrixLinkLattice<math::group::SU2, T>& out,
                                    double scale,
                                    std::size_t base,
                                    std::size_t cnt) noexcept {
        constexpr std::size_t k_batch = math::group::k_gauge_batch<T>;
        auto const& sh                = u.shape();
        std::size_t const d           = u.ndims();
        std::size_t const span        = u.link_span();  // padded component stride
        std::size_t const l0          = sh[0];
        T const scl                   = static_cast<T>(scale);

        std::size_t stg[4] = {1, 0, 0, 0};  // strides: stg[j] = ∏_{k<j} L[k]
        for (std::size_t j = 1; j < d; ++j) {
            stg[j] = stg[j - 1] * sh[j - 1];
        }
        std::size_t const r0    = base / l0;
        std::size_t const nrows = cnt / l0;

        for (std::size_t r = r0; r < r0 + nrows; ++r) {
            std::size_t const row = r * l0;
            std::size_t fwd[4]    = {0, 0, 0, 0};
            std::size_t bwd[4]    = {0, 0, 0, 0};
            bool fwd_wrap[4]      = {false, false, false, false};
            bool bwd_wrap[4]      = {false, false, false, false};
            std::size_t rr        = r;
            for (std::size_t j = 1; j < d; ++j) {
                std::size_t const lj = sh[j];
                std::size_t const cj = rr % lj;
                rr /= lj;
                fwd[j]      = (cj + 1 < lj) ? stg[j] : (lj - 1) * stg[j];
                fwd_wrap[j] = (cj + 1 == lj);
                bwd[j]      = (cj > 0) ? stg[j] : (lj - 1) * stg[j];
                bwd_wrap[j] = (cj == 0);
            }
            // Dir-0 neighbours are relative to the passed site (see SU3 note):
            // s_pmu_mnu shifts s+μ̂, which is in a different row when μ ≥ 1.
            auto nxt = [&](std::size_t s, std::size_t x, std::size_t mu) -> std::size_t {
                if (mu == 0) {
                    return (x + 1 == l0) ? s - (l0 - 1) : s + 1;
                }
                return fwd_wrap[mu] ? s - fwd[mu] : s + fwd[mu];
            };
            auto prv = [&](std::size_t s, std::size_t x, std::size_t mu) -> std::size_t {
                if (mu == 0) {
                    return (x == 0) ? s + (l0 - 1) : s - 1;
                }
                return bwd_wrap[mu] ? s + bwd[mu] : s - bwd[mu];
            };

            for (std::size_t mu = 0; mu < d; ++mu) {
                T const* const u_mu_blk = u.mu_block_data(mu);
                T* const out_mu_blk     = out.mu_block_data(mu);

                std::size_t x = 0;
                for (; x + k_batch <= l0; x += k_batch) {
                    std::size_t const s_base = row + x;
                    std::size_t s_pmu[k_batch];
                    for (std::size_t b = 0; b < k_batch; ++b) {
                        s_pmu[b] = nxt(s_base + b, x + b, mu);
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
                            std::size_t const xb = x + b;
                            s_pnu[b]             = nxt(s_base + b, xb, nu);
                            s_mnu[b]             = prv(s_base + b, xb, nu);
                            s_pmu_mnu[b]         = prv(s_pmu[b], xb, nu);
                        }

                        // -------- Forward staple ------------------------------
                        T a_re[4][k_batch];
                        T a_im[4][k_batch];
                        T b_re[4][k_batch];
                        T b_im[4][k_batch];
                        T c_re[4][k_batch];
                        T c_im[4][k_batch];
                        math::group::load_links_batched(a_re, a_im, u_nu_blk, span, s_pmu);
                        math::group::load_links_batched(b_re, b_im, u_mu_blk, span, s_pnu);
                        math::group::load_links_batched(c_re, c_im, u_nu_blk, span, s_base);

                        // t1 = a · b† ,  v += t1 · c†
                        T t1_re[4][k_batch];
                        T t1_im[4][k_batch];
                        math::su2::mul_adj_2x2_batched<false>(t1_re, t1_im, a_re, a_im, b_re, b_im);
                        math::su2::mul_adj_2x2_batched<true>(v_re, v_im, t1_re, t1_im, c_re, c_im);

                        // -------- Backward staple -----------------------------
                        math::group::load_links_batched(a_re, a_im, u_nu_blk, span, s_pmu_mnu);
                        math::group::load_links_batched(b_re, b_im, u_mu_blk, span, s_mnu);
                        math::group::load_links_batched(c_re, c_im, u_nu_blk, span, s_mnu);
                        // t1 = b† · c ,  v += a† · t1
                        math::su2::adj_mul_2x2_batched<false>(t1_re, t1_im, b_re, b_im, c_re, c_im);
                        math::su2::adj_mul_2x2_batched<true>(v_re, v_im, a_re, a_im, t1_re, t1_im);
                    }

                    // ------------ Final: U_μ(s) · V → TA → scatter ------------
                    T u_re[4][k_batch];
                    T u_im[4][k_batch];
                    math::group::load_links_batched(u_re, u_im, u_mu_blk, span, s_base);
                    // uv = U · V, then TA into the algebra.
                    T uv_re[4][k_batch];
                    T uv_im[4][k_batch];
                    math::su2::mul_2x2_batched(uv_re, uv_im, u_re, u_im, v_re, v_im);
                    T ta_re[4][k_batch];
                    T ta_im[4][k_batch];
                    math::su2::traceless_antiherm_2x2_batched(ta_re, ta_im, uv_re, uv_im);
                    for (std::size_t k = 0; k < 4; ++k) {
                        std::size_t const off_re = (2 * k) * span;
                        std::size_t const off_im = ((2 * k) + 1) * span;
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
                // Row remainder (l0 % k_batch sites) — scalar, same math, same
                // strided neighbours. Nonzero only when k_batch ∤ L0.
                for (; x < l0; ++x) {
                    force_one_site_<Fused>(
                        u, u_mu_blk, out_mu_blk, scale, mu, d, span, row + x, x, nxt, prv);
                }
            }
        }
    }

private:
    // One link's staple force — the cold scalar path for the per-row remainder.
    // Templated on the row's strided next/prev callables so the neighbour
    // addressing matches the batched path exactly. Same 8-real math, per site.
    template <bool Fused, class T, class Nxt, class Prv>
    static void force_one_site_(MatrixLinkLattice<math::group::SU2, T> const& u,
                                T const* u_mu_blk,
                                T* out_mu_blk,
                                double scale,
                                std::size_t mu,
                                std::size_t d,
                                std::size_t span,
                                std::size_t s,
                                std::size_t x,
                                Nxt const& nxt,
                                Prv const& prv) noexcept {
        auto load_link = [span](double* dst, auto const* blk, std::size_t site) noexcept {
            for (std::size_t k = 0; k < 8; ++k) {
                dst[k] = static_cast<double>(blk[(k * span) + site]);
            }
        };
        double v[8]             = {0, 0, 0, 0, 0, 0, 0, 0};
        std::size_t const s_pmu = nxt(s, x, mu);
        for (std::size_t nu = 0; nu < d; ++nu) {
            if (nu == mu) {
                continue;
            }
            T const* const u_nu_blk     = u.mu_block_data(nu);
            std::size_t const s_pnu     = nxt(s, x, nu);
            std::size_t const s_mnu     = prv(s, x, nu);
            std::size_t const s_pmu_mnu = prv(s_pmu, x, nu);
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
                out_mu_blk[(k * span) + s] += static_cast<T>(scale * ta[k]);
            }
        } else {
            for (std::size_t k = 0; k < 8; ++k) {
                out_mu_blk[(k * span) + s] = static_cast<T>(scale * ta[k]);
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
