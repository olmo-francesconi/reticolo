#pragma once

#include <reticolo/action/detail/gauge_group/base.hpp>
#include <reticolo/core/indexing.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/math/su2_ops.hpp>

#include <cstddef>

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
    template <class T>
    static void compute_force(MatrixLinkLattice<SU2, T> const& u,
                              MatrixLinkLattice<SU2, T>& force,
                              double beta_over_n) noexcept {
        std::size_t const d   = u.ndims();
        std::size_t const ns  = u.nsites();
        Indexing const& idx   = u.indexing_ref();
        double const neg_b_oN = -beta_over_n;

        auto load_link = [ns](double* dst, T const* blk, std::size_t s) noexcept {
            for (std::size_t k = 0; k < 8; ++k) {
                dst[k] = static_cast<double>(blk[(k * ns) + s]);
            }
        };

        for (std::size_t mu = 0; mu < d; ++mu) {
            T const* const u_mu_blk = u.mu_block_data(mu);
            T* const f_mu_blk       = force.mu_block_data(mu);
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
                for (std::size_t k = 0; k < 8; ++k) {
                    f_mu_blk[(k * ns) + s] = static_cast<T>(neg_b_oN * ta[k]);
                }
            }
        }
    }
};

static_assert(GaugeGroup<SU2>);

}  // namespace reticolo::gauge_group
