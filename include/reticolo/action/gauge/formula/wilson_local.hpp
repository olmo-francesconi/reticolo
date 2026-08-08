#pragma once

#include <reticolo/action/concepts.hpp>
#include <reticolo/core/exec/nn_checkerboard.hpp>
#include <reticolo/core/field/matrix_link_lattice.hpp>
#include <reticolo/core/field/site.hpp>
#include <reticolo/math/group/formula/su2_formula.hpp>
#include <reticolo/math/group/formula/su3_formula.hpp>
#include <reticolo/math/group/su2.hpp>
#include <reticolo/math/group/su3.hpp>
#include <reticolo/math/group/u1.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

// Wilson LOCAL update — the staple gather and the per-link Metropolis move that
// `Wilson<G>::metropolis_stencil` runs, the gauge peer of `wilson_kernels<G>`'s
// force. It lives in the ACTION layer, not in `math/group/`, for the reason the
// force does: the plaquette is Wilson's physics, while `G` carries only the group
// constants and the algebra hooks.
//
// The move. For link U_mu(x), every plaquette containing it factorises as
// Re Tr[U_mu(x)·A] with the STAPLE
//
//   A = Σ_{ν≠μ} [ U_ν(x+μ̂)  U_μ(x+ν̂)† U_ν(x)†                (forward)
//               + U_ν(x+μ̂−ν̂)† U_μ(x−ν̂)† U_ν(x−ν̂) ]           (backward)
//
// so the local action is −(β/N)·Re Tr[U_μ(x)·A] and a move U → U' = exp(σH)·U
// costs ΔS = −(β/N)·(Re Tr[U'A] − Re Tr[UA]) with ONE staple gather serving both.
//
// H is the algebra element the updater already drew into the noise field — the
// SAME `sample_algebra_slab` output HMC uses for momenta, exponentiated with the
// SAME closed-form `exp_su2` / `exp_su3` the integrator's drift uses. So the
// proposal costs no new group math: a near-identity element with the correct
// Haar-symmetric distribution falls out of machinery that already existed.
//
// COLOURING is (direction, site parity), 2·ndims passes: every link in the staple
// of (x,μ) is either of a different direction or on the opposite parity, so one
// pass writes links nothing in that pass reads.
//
// This is the correct, readable form: per-link gathers through `next`/`prev` and
// compact matrix temporaries. It is NOT fused with `compute_force_range`'s
// vectorised plane walk, which sweeps whole rows with hoisted neighbour bases —
// that fusion is the optimisation, and it wants a benchmark before it is worth
// the duplication.

namespace reticolo::action::formula {

// Per-group matrix ops for the local update, in a uniform MATRIX representation
// (N×N complex as 2N² reals) so the sweep below is written once. The storage
// representation is a separate thing: SU(N) stores the matrix directly, U(1)
// stores the angle, so `load`/`store` are where they differ.
template <class G>
struct gauge_local_ops;

template <>
struct gauge_local_ops<math::group::SU2> {
    static constexpr std::size_t n_mat = 8;  // 2×2 complex
    template <class T>
    static void load(double* out, T const* blk, std::size_t span, std::size_t s) noexcept {
        for (std::size_t k = 0; k < n_mat; ++k) {
            out[k] = static_cast<double>(blk[(k * span) + s]);
        }
    }
    template <class T>
    static void store(T* blk, std::size_t span, std::size_t s, double const* in) noexcept {
        for (std::size_t k = 0; k < n_mat; ++k) {
            blk[(k * span) + s] = static_cast<T>(in[k]);
        }
    }
    static void mm(double* o, double const* a, double const* b) noexcept {
        math::su2::mul_2x2<double>(o, a, b);
    }
    static void mmd(double* o, double const* a, double const* b) noexcept {
        math::su2::mul_adj_2x2(o, a, b);
    }
    static void mdm(double* o, double const* a, double const* b) noexcept {
        math::su2::adj_mul_2x2(o, a, b);
    }
    // Re Tr(a·b) = Σ_ij Re(a_ij · b_ji) — no full product needed.
    [[nodiscard]] static double re_tr_mm(double const* a, double const* b) noexcept {
        double t = 0.0;
        for (std::size_t i = 0; i < 2; ++i) {
            for (std::size_t j = 0; j < 2; ++j) {
                std::size_t const aij = 2 * ((i * 2) + j);
                std::size_t const bji = 2 * ((j * 2) + i);
                t += (a[aij] * b[bji]) - (a[aij + 1] * b[bji + 1]);
            }
        }
        return t;
    }
    // out = exp(dt·H)·U, H the algebra element in storage layout.
    template <class T>
    static void exp_mul(double* out,
                        T const* h_blk,
                        std::size_t span,
                        std::size_t s,
                        double dt,
                        double const* u) noexcept {
        double h[n_mat];
        double v[n_mat];
        load(h, h_blk, span, s);
        math::su2::exp_su2(v, h, dt);
        mm(out, v, u);
    }
};

template <>
struct gauge_local_ops<math::group::SU3> {
    static constexpr std::size_t n_mat = 18;  // 3×3 complex
    template <class T>
    static void load(double* out, T const* blk, std::size_t span, std::size_t s) noexcept {
        for (std::size_t k = 0; k < n_mat; ++k) {
            out[k] = static_cast<double>(blk[(k * span) + s]);
        }
    }
    template <class T>
    static void store(T* blk, std::size_t span, std::size_t s, double const* in) noexcept {
        for (std::size_t k = 0; k < n_mat; ++k) {
            blk[(k * span) + s] = static_cast<T>(in[k]);
        }
    }
    static void mm(double* o, double const* a, double const* b) noexcept {
        math::su3::mul_3x3(o, a, b);
    }
    static void mmd(double* o, double const* a, double const* b) noexcept {
        math::su3::mul_adj_3x3(o, a, b);
    }
    static void mdm(double* o, double const* a, double const* b) noexcept {
        math::su3::adj_mul_3x3(o, a, b);
    }
    [[nodiscard]] static double re_tr_mm(double const* a, double const* b) noexcept {
        double t = 0.0;
        for (std::size_t i = 0; i < 3; ++i) {
            for (std::size_t j = 0; j < 3; ++j) {
                std::size_t const aij = 2 * ((i * 3) + j);
                std::size_t const bji = 2 * ((j * 3) + i);
                t += (a[aij] * b[bji]) - (a[aij + 1] * b[bji + 1]);
            }
        }
        return t;
    }
    template <class T>
    static void exp_mul(double* out,
                        T const* h_blk,
                        std::size_t span,
                        std::size_t s,
                        double dt,
                        double const* u) noexcept {
        double h[n_mat];
        double v[n_mat];
        load(h, h_blk, span, s);
        math::su3::exp_su3(v, h, dt);
        mm(out, v, u);
    }
};

// U(1): stored as the angle θ, but the SAME sweep runs on the 1×1 unitary matrix
// e^{iθ} = (cos θ, sin θ). Staple sums leave the group in either case — for SU(N)
// too — so a complex accumulator is the natural representation, and `store`
// projects back to an angle.
template <>
struct gauge_local_ops<math::group::U1> {
    static constexpr std::size_t n_mat = 2;  // 1×1 complex
    template <class T>
    static void load(double* out, T const* blk, std::size_t span, std::size_t s) noexcept {
        (void)span;
        auto const th = static_cast<double>(blk[s]);
        out[0]        = std::cos(th);
        out[1]        = std::sin(th);
    }
    template <class T>
    static void store(T* blk, std::size_t span, std::size_t s, double const* in) noexcept {
        (void)span;
        blk[s] = static_cast<T>(std::atan2(in[1], in[0]));
    }
    static void mm(double* o, double const* a, double const* b) noexcept {
        o[0] = (a[0] * b[0]) - (a[1] * b[1]);
        o[1] = (a[0] * b[1]) + (a[1] * b[0]);
    }
    static void mmd(double* o, double const* a, double const* b) noexcept {
        o[0] = (a[0] * b[0]) + (a[1] * b[1]);
        o[1] = (a[1] * b[0]) - (a[0] * b[1]);
    }
    static void mdm(double* o, double const* a, double const* b) noexcept {
        o[0] = (a[0] * b[0]) + (a[1] * b[1]);
        o[1] = (a[0] * b[1]) - (a[1] * b[0]);
    }
    [[nodiscard]] static double re_tr_mm(double const* a, double const* b) noexcept {
        return (a[0] * b[0]) - (a[1] * b[1]);
    }
    template <class T>
    static void exp_mul(double* out,
                        T const* h_blk,
                        std::size_t span,
                        std::size_t s,
                        double dt,
                        double const* u) noexcept {
        (void)span;
        double const a = dt * static_cast<double>(h_blk[s]);
        double const r[2]{std::cos(a), std::sin(a)};
        mm(out, r, u);
    }
};

// The five sites a link's staple reads, as flat indices. Every one is a hoisted
// ROW BASE plus an in-row x offset — no division survives into the inner loop.
// `Indexing::next`/`prev` compute a coordinate with an integer divide per call,
// which is fine for a cold path and ruinous here: an SU(3) 4D link would pay ~13
// of them per site. Row bases are derived once per row instead, and the only
// x-dependent shift is the ±1 wrap on the contiguous axis.
struct StapleRows_ {
    std::size_t row_pmu = 0;                   // row + μ̂   (μ ≥ 1; μ = 0 is an x shift)
    std::array<std::size_t, 4> row_pnu{};      // row + ν̂
    std::array<std::size_t, 4> row_mnu{};      // row − ν̂
    std::array<std::size_t, 4> row_pmu_mnu{};  // row + μ̂ − ν̂
};

// One (direction, parity) colour of a Wilson Metropolis sweep. Updates the links
// of direction `mu` on sites of parity `parity`, in place.
//
// Threading is the field's canonical partition, folded in canonical item order —
// so the totals are deterministic for a fixed (team, slabs) config, exactly like
// every other gauge pass. Write-disjointness holds because a staple of (x,μ) only
// reaches links of another direction or of the opposite parity.
template <class G, class T>
[[nodiscard]] inline LocalStats wilson_metropolis_stencil(MatrixLinkLattice<G, T>& u,
                                                          std::size_t mu,
                                                          int parity,
                                                          MatrixLinkLattice<G, T> const& noise,
                                                          Lattice<real_scalar_t<T>> const& logu,
                                                          double sigma,
                                                          double beta_over_n) noexcept {
    using ops                = gauge_local_ops<G>;
    constexpr std::size_t nm = ops::n_mat;

    auto const& sh         = u.shape();
    std::size_t const d    = u.ndims();
    std::size_t const l0   = sh[0];
    std::size_t const span = u.link_span();
    T* const u_mu          = u.mu_block_data(mu);
    T const* const h_mu    = noise.mu_block_data(mu);
    auto const* const lu   = logu.data();
    bool const mu_is_x     = (mu == 0);

    std::array<std::size_t, 4> stride{};
    stride[0] = 1;
    for (std::size_t k = 1; k < d; ++k) {
        stride[k] = stride[k - 1] * sh[k - 1];
    }
    std::array<T const*, 4> u_nu_blk{};
    for (std::size_t nu = 0; nu < d; ++nu) {
        u_nu_blk[nu] = u.mu_block_data(nu);
    }

    exec::Partition const p = exec::partition(u);
    int const nthreads      = exec::traverse_threads(p.n_sites, u.bytes_per_site());

    return exec::parallel_reduce<LocalStats>(nthreads, p.n_items, [&](std::size_t it) {
        std::size_t const base = it * p.item_sites;
        std::size_t const cnt  = std::min(p.item_sites, p.n_sites - base);
        LocalStats acc{};

        for (std::size_t row = base; row < base + cnt; row += l0) {
            // ---- per-row geometry, computed once ----------------------------
            std::array<std::size_t, 4> coord{};
            std::size_t par = 0;
            for (std::size_t k = 1; k < d; ++k) {
                coord[k] = (row / stride[k]) % sh[k];
                par += coord[k];
            }
            // Shift a base that already carries `row`'s coordinate in axis k.
            auto const shift = [&](std::size_t b, std::size_t k, bool fwd) {
                if (fwd) {
                    return (coord[k] + 1 == sh[k]) ? (b - ((sh[k] - 1) * stride[k]))
                                                   : (b + stride[k]);
                }
                return (coord[k] == 0) ? (b + ((sh[k] - 1) * stride[k])) : (b - stride[k]);
            };
            StapleRows_ r{};
            r.row_pmu = mu_is_x ? row : shift(row, mu, true);
            for (std::size_t nu = 0; nu < d; ++nu) {
                if (nu == mu) {
                    continue;
                }
                r.row_pnu[nu]     = (nu == 0) ? row : shift(row, nu, true);
                r.row_mnu[nu]     = (nu == 0) ? row : shift(row, nu, false);
                r.row_pmu_mnu[nu] = (nu == 0) ? r.row_pmu : shift(r.row_pmu, nu, false);
            }

            std::size_t const x0 = (par + static_cast<std::size_t>(parity)) & 1U;
            for (std::size_t x = x0; x < l0; x += 2) {
                std::size_t const s     = row + x;
                std::size_t const xp    = (x + 1 == l0) ? 0 : (x + 1);
                std::size_t const xm    = (x == 0) ? (l0 - 1) : (x - 1);
                std::size_t const s_pmu = mu_is_x ? (row + xp) : (r.row_pmu + x);

                double staple[nm]{};
                double a[nm];
                double b[nm];
                double c[nm];

                for (std::size_t nu = 0; nu < d; ++nu) {
                    if (nu == mu) {
                        continue;
                    }
                    T const* const u_nu     = u_nu_blk[nu];
                    bool const nu_is_x      = (nu == 0);
                    std::size_t const s_pnu = nu_is_x ? (row + xp) : (r.row_pnu[nu] + x);
                    std::size_t const s_mnu = nu_is_x ? (row + xm) : (r.row_mnu[nu] + x);
                    // s + μ̂ − ν̂. Whichever of μ, ν is the contiguous axis supplies
                    // the x shift and the other supplies the row; μ ≠ ν, so at most
                    // one of them is axis 0.
                    std::size_t pmu_mnu_row = r.row_pmu_mnu[nu];
                    std::size_t pmu_mnu_x   = x;
                    if (mu_is_x) {
                        pmu_mnu_row = r.row_mnu[nu];
                        pmu_mnu_x   = xp;
                    } else if (nu_is_x) {
                        pmu_mnu_row = r.row_pmu;
                        pmu_mnu_x   = xm;
                    }
                    std::size_t const s_pmu_mnu = pmu_mnu_row + pmu_mnu_x;

                    // forward: U_ν(s+μ̂) · U_μ(s+ν̂)† · U_ν(s)†
                    ops::load(a, u_nu, span, s_pmu);
                    ops::load(b, u_mu, span, s_pnu);
                    ops::mmd(c, a, b);
                    ops::load(b, u_nu, span, s);
                    ops::mmd(a, c, b);
                    for (std::size_t k = 0; k < nm; ++k) {
                        staple[k] += a[k];
                    }

                    // backward: U_ν(s+μ̂−ν̂)† · U_μ(s−ν̂)† · U_ν(s−ν̂)
                    //         = [U_μ(s−ν̂) · U_ν(s+μ̂−ν̂)]† · U_ν(s−ν̂)
                    ops::load(a, u_mu, span, s_mnu);
                    ops::load(b, u_nu, span, s_pmu_mnu);
                    ops::mm(c, a, b);
                    ops::load(b, u_nu, span, s_mnu);
                    ops::mdm(a, c, b);
                    for (std::size_t k = 0; k < nm; ++k) {
                        staple[k] += a[k];
                    }
                }

                // ---- propose, score, accept ---------------------------------
                double link[nm];
                double prop[nm];
                ops::load(link, u_mu, span, s);
                ops::exp_mul(prop, h_mu, span, s, sigma, link);

                double const ds =
                    -beta_over_n * (ops::re_tr_mm(prop, staple) - ops::re_tr_mm(link, staple));
                if (-ds >= static_cast<double>(lu[s])) {
                    ops::store(u_mu, span, s, prop);
                    acc.n_accepted += 1;
                    acc.ds += ds;
                }
                acc.n_attempts += 1;
            }
        }
        return acc;
    });
}

}  // namespace reticolo::action::formula
