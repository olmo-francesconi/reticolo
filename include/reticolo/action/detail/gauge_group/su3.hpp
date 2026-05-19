#pragma once

#include <reticolo/action/detail/gauge_group/base.hpp>
#include <reticolo/core/indexing.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/math/su3_ops.hpp>

#include <cstddef>

namespace reticolo::gauge_group {

// =============================================================================
//  SU(3) gauge group model. Storage: full 3×3 complex matrix, 18 reals per
//  link element (see math::su3 for the layout). N = 3, so the Wilson
//  prefactor is β/3 and the per-link constant in s_full is 3·n_plaq.
//
//  Same surface as `SU2`: GaugeGroup concept members + per-plaquette
//  `plaq_re_tr` + HMC slab hooks (sample/kinetic/expi_lmul/project) + a
//  link-centric `compute_force` that walks each (x, μ) and assembles the
//  full staple sum + TA[U·V] in one pass.
// =============================================================================

struct SU3 {
    using scalar_t                                 = double;
    static constexpr std::size_t n_real_components = 18;
    static constexpr std::size_t n_color           = 3;

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
    template <class Rng>
    [[gnu::always_inline]] static inline void
    sample_algebra_slab(double* p_blk, Rng& rng, std::size_t n) noexcept {
        math::su3::sample_algebra_slab(p_blk, rng, n);
    }

    [[gnu::always_inline]] static inline double kinetic_slab(double const* p_blk,
                                                             std::size_t n) noexcept {
        return math::su3::kinetic_slab(p_blk, n);
    }

    [[gnu::always_inline]] static inline void
    expi_lmul_slab(double* u_blk, double const* p_blk, double dt, std::size_t n) noexcept {
        math::su3::expi_lmul_slab(u_blk, p_blk, dt, n);
    }

    // -------- link-centric Wilson force --------------------------------------
    // Same shape as SU2::compute_force, just with 18-real matrices.
    //   V_μ(x) = sum_{ν ≠ μ} [ fwd_μν(x) + bwd_μν(x) ]
    //   F_μ(x) = −(β/N) · TA[U_μ(x) · V_μ(x)]
    template <class T>
    static void compute_force(MatrixLinkLattice<SU3, T> const& u,
                              MatrixLinkLattice<SU3, T>& force,
                              double beta_over_n) noexcept {
        std::size_t const d   = u.ndims();
        std::size_t const ns  = u.nsites();
        Indexing const& idx   = u.indexing_ref();
        double const neg_b_oN = -beta_over_n;

        auto load_link = [ns](double* dst, T const* blk, std::size_t s) noexcept {
            for (std::size_t k = 0; k < 18; ++k) {
                dst[k] = static_cast<double>(blk[(k * ns) + s]);
            }
        };

        for (std::size_t mu = 0; mu < d; ++mu) {
            T const* const u_mu_blk = u.mu_block_data(mu);
            T* const f_mu_blk       = force.mu_block_data(mu);
            for (std::size_t s = 0; s < ns; ++s) {
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

                    // Forward staple: U_ν(s+μ) · U_μ(s+ν)† · U_ν(s)†
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
                    // Backward staple: U_ν(s+μ−ν)† · U_μ(s−ν)† · U_ν(s−ν)
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
                for (std::size_t k = 0; k < 18; ++k) {
                    f_mu_blk[(k * ns) + s] = static_cast<T>(neg_b_oN * ta[k]);
                }
            }
        }
    }
};

static_assert(GaugeGroup<SU3>);

}  // namespace reticolo::gauge_group
