#pragma once

#include <reticolo/action/gauge/formula/wilson_kernels.hpp>
#include <reticolo/core/indexing.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/parallel.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/math/group/su3.hpp>
#include <reticolo/math/su3_ops.hpp>

#include <cstddef>

// Wilson plaquette kernels for SU(3) — the action-specific physics (Re Tr U_p,
// the Σ Re Tr plane fast-path, the link-centric staple force / fused kick) for
// the Wilson action on a MatrixLinkLattice<math::group::SU3, T>. Kept out of the SU3 group
// model, which holds only the core group operations.

namespace reticolo::action::formula {

template <>
struct wilson_kernels<math::group::SU3> {
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
    // Σ Re Tr U_p over the sites [base, base+cnt) of one (μ, ν) plane. Pure per-
    // range worker — no threading. Table-free: the two plaquette-forward
    // neighbours (s+μ̂, s+ν̂) are computed from the lattice strides by the same
    // row-nested sweep the force uses, so the gauge s_full no longer reads the
    // Indexing tables. The partition hands whole rows (base, cnt multiples of L0).
    template <class T>
    static double s_full_plane_range(MatrixLinkLattice<math::group::SU3, T> const& u,
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
                T a_re[9][k_batch];
                T a_im[9][k_batch];
                T b_re[9][k_batch];
                T b_im[9][k_batch];
                T c_re[9][k_batch];
                T c_im[9][k_batch];
                T d_re[9][k_batch];
                T d_im[9][k_batch];
                math::group::load_links_batched(a_re, a_im, u_mu_blk, span, s_base);
                math::group::load_links_batched(b_re, b_im, u_nu_blk, span, s_pmu);
                math::group::load_links_batched(c_re, c_im, u_mu_blk, span, s_pnu);
                math::group::load_links_batched(d_re, d_im, u_nu_blk, span, s_base);
                T ab_re[9][k_batch];
                T ab_im[9][k_batch];
                T dc_re[9][k_batch];
                T dc_im[9][k_batch];
                math::su3::mul_3x3_batched(ab_re, ab_im, a_re, a_im, b_re, b_im);
                math::su3::mul_3x3_batched(dc_re, dc_im, d_re, d_im, c_re, c_im);
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
            for (; x < l0; ++x) {
                std::size_t const s = row + x;
                total += plaq_re_tr(u_mu_blk, u_nu_blk, s, nxt(s, x, mu), nxt(s, x, nu), span);
            }
        }
        return total;
    }

    // -------- link-centric Wilson force --------------------------------------
    // Same shape as SU2::compute_force, just with 18-real matrices.
    //   V_μ(x) = sum_{ν ≠ μ} [ fwd_μν(x) + bwd_μν(x) ]
    //   F_μ(x) = −(β/N) · TA[U_μ(x) · V_μ(x)]
    //
    // Body shared between the plain force (out = F) and the fused kick
    // (mom += k_dt · F) via a templated `Fused` non-type parameter.
public:
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
    static void compute_force_range(MatrixLinkLattice<math::group::SU3, T> const& u,
                                    MatrixLinkLattice<math::group::SU3, T>& out,
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
            // Per-row |offset| to the ±ĵ neighbour for each direction j ≥ 1 (dir 0
            // is per-x). `*wrap` flags that the offset crosses the periodic seam,
            // so it is subtracted (fwd) / added (bwd) instead — matching the
            // Indexing table's wrap exactly.
            std::size_t fwd[4] = {0, 0, 0, 0};
            std::size_t bwd[4] = {0, 0, 0, 0};
            bool fwd_wrap[4]   = {false, false, false, false};
            bool bwd_wrap[4]   = {false, false, false, false};
            std::size_t rr     = r;
            for (std::size_t j = 1; j < d; ++j) {
                std::size_t const lj = sh[j];
                std::size_t const cj = rr % lj;
                rr /= lj;
                fwd[j]      = (cj + 1 < lj) ? stg[j] : (lj - 1) * stg[j];
                fwd_wrap[j] = (cj + 1 == lj);
                bwd[j]      = (cj > 0) ? stg[j] : (lj - 1) * stg[j];
                bwd_wrap[j] = (cj == 0);
            }
            // Dir-0 neighbours are expressed relative to the passed site s (not
            // the row base): s_pmu_mnu shifts s+μ̂, which lives in a DIFFERENT row
            // when μ ≥ 1, so the x-wrap must key off that site, not the current row.
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
                            std::size_t const xb = x + b;
                            s_pnu[b]             = nxt(s_base + b, xb, nu);
                            s_mnu[b]             = prv(s_base + b, xb, nu);
                            s_pmu_mnu[b]         = prv(s_pmu[b], xb, nu);
                        }

                        // ------------- Forward staple ---------------------------
                        // t1 = U_ν(s+μ̂) · U_μ(s+ν̂)† ,  t2 = t1 · U_ν(s)† ,  v += t2
                        T a_re[9][k_batch];
                        T a_im[9][k_batch];
                        T b_re[9][k_batch];
                        T b_im[9][k_batch];
                        T c_re[9][k_batch];
                        T c_im[9][k_batch];
                        math::group::load_links_batched(a_re, a_im, u_nu_blk, span, s_pmu);
                        math::group::load_links_batched(b_re, b_im, u_mu_blk, span, s_pnu);
                        math::group::load_links_batched(c_re, c_im, u_nu_blk, span, s_base);

                        // t1 = a · b† ,  v += t1 · c†
                        T t1_re[9][k_batch];
                        T t1_im[9][k_batch];
                        math::su3::mul_adj_3x3_batched<false>(t1_re, t1_im, a_re, a_im, b_re, b_im);
                        math::su3::mul_adj_3x3_batched<true>(v_re, v_im, t1_re, t1_im, c_re, c_im);

                        // ------------- Backward staple --------------------------
                        // t1 = U_μ(s-ν̂)† · U_ν(s-ν̂) ,  t2 = U_ν(s+μ̂-ν̂)† · t1
                        math::group::load_links_batched(a_re, a_im, u_nu_blk, span, s_pmu_mnu);
                        math::group::load_links_batched(b_re, b_im, u_mu_blk, span, s_mnu);
                        math::group::load_links_batched(c_re, c_im, u_nu_blk, span, s_mnu);
                        // t1 = b† · c ,  v += a† · t1
                        math::su3::adj_mul_3x3_batched<false>(t1_re, t1_im, b_re, b_im, c_re, c_im);
                        math::su3::adj_mul_3x3_batched<true>(v_re, v_im, a_re, a_im, t1_re, t1_im);
                    }

                    // ------------ Final: U_μ(s) · V → TA → scatter ------------
                    T u_re[9][k_batch];
                    T u_im[9][k_batch];
                    math::group::load_links_batched(u_re, u_im, u_mu_blk, span, s_base);
                    // uv = U · V, then TA into the algebra.
                    T uv_re[9][k_batch];
                    T uv_im[9][k_batch];
                    math::su3::mul_3x3_batched(uv_re, uv_im, u_re, u_im, v_re, v_im);
                    T ta_re[9][k_batch];
                    T ta_im[9][k_batch];
                    math::su3::traceless_antiherm_3x3_batched(ta_re, ta_im, uv_re, uv_im);
                    // Scatter scale·ta into out_mu_blk.
                    for (std::size_t k = 0; k < 9; ++k) {
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
    // addressing matches the batched path exactly. Same 18-real math as the
    // batched staple, per site.
    template <bool Fused, class T, class Nxt, class Prv>
    static void force_one_site_(MatrixLinkLattice<math::group::SU3, T> const& u,
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
            for (std::size_t k = 0; k < 18; ++k) {
                dst[k] = static_cast<double>(blk[(k * span) + site]);
            }
        };
        double v[18]            = {};
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
                out_mu_blk[(k * span) + s] += static_cast<T>(scale * ta[k]);
            }
        } else {
            for (std::size_t k = 0; k < 18; ++k) {
                out_mu_blk[(k * span) + s] = static_cast<T>(scale * ta[k]);
            }
        }
    }

public:
    // Precision-generic: the batched kernel is T-native (float → 4-/8-wide,
    // double → 2-wide) and folds the ns % k_batch remainder into the per-site
    // tail, so both precisions take the same path. The batched kernel writes
    // each (μ, s) exactly once, so the pass is a write-disjoint map — `field_visit`
    // worksplits it over k_batch-aligned site chunks, bit-identical to the serial
    // whole-field sweep for any thread count (no force zero-init needed).
    template <class T>
    static void compute_force(MatrixLinkLattice<math::group::SU3, T> const& u,
                              MatrixLinkLattice<math::group::SU3, T>& force,
                              double beta_over_n) noexcept {
        reticolo::exec::field_visit(
            u, math::group::k_gauge_batch<T>, [&](std::size_t base, std::size_t cnt) {
                compute_force_range<false>(u, force, -beta_over_n, base, cnt);
            });
    }

    template <class T>
    static void compute_force_and_kick(MatrixLinkLattice<math::group::SU3, T> const& u,
                                       MatrixLinkLattice<math::group::SU3, T>& mom,
                                       double beta_over_n,
                                       double k_dt) noexcept {
        reticolo::exec::field_visit(
            u, math::group::k_gauge_batch<T>, [&](std::size_t base, std::size_t cnt) {
                compute_force_range<true>(u, mom, -k_dt * beta_over_n, base, cnt);
            });
    }
};

}  // namespace reticolo::action::formula
