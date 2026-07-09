#pragma once

#include <reticolo/action/gauge/formula/wilson_kernels.hpp>
#include <reticolo/action/gauge/formula/wilson_u1_formula.hpp>
#include <reticolo/core/indexing.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/parallel.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/math/group/u1.hpp>
#include <reticolo/math/vec_libm.hpp>

#include <cstddef>
#include <vector>

// Wilson plaquette kernels for U(1) — the action-specific physics for the
// Wilson action on a `MatrixLinkLattice<U1, T>` (the abelian angle field).
//
// The force is LINK-CENTRIC (write-once per link), derived from the
// plaquette-scatter identity by regrouping each link's scatter contributions
// by destination link instead of by source plaquette:
//
//   F_mu(x) = -(beta/N) * sum_{nu != mu} sgn(mu,nu) *
//               [ sinP_{plane(mu,nu)}(x) - sinP_{plane(mu,nu)}(prev(x,nu)) ]
//
// where plane(mu,nu) = (min(mu,nu), max(mu,nu)) and sgn(mu,nu) = +1 if
// mu < nu else -1 (P is antisymmetric under mu<->nu: sin P_{mu,nu} =
// -sin P_{nu,mu}). Two-phase, both on the canonical field partition
// (`exec::field_visit` / `field_reduce`) so the whole pass worksplits with the
// same thread↔slab ownership as every other op:
//
//   Phase 1 (fill) — one Sleef-batched sin(angle) pass per (mu, nu) plane
//     (mu < nu), stored into a persistent `nplanes*ns` scratch.
//   Phase 2 (gather) — each link (mu, x) reads only its own plane's sinP
//     entries at x and at `prev(x, nu)` — never anyone else's link — and
//     writes exactly once, so a chunk over x is write-disjoint.
//
// Precision-generic: the batched transcendentals always run in `double`
// regardless of the field's T (cast in on read, cast out on write) — one
// path for every T, no separate scalar fallback.

namespace reticolo::action::formula {

namespace impl {

// Thread-local scratch shared by every U(1) Wilson kernel entry point below.
// Each entry point calls this exactly once per invocation, with the total
// size it needs, and slices sub-spans out of the single pointer — so a
// resize inside one call can never invalidate a pointer live in another part
// of the same call.
[[gnu::always_inline]] inline double* plane_scratch(std::size_t n) noexcept {
    thread_local std::vector<double> buf;
    return reticolo::exec::thread_scratch(buf, n);
}

// Canonical running index of plane (a, b), a < b, in the a-major
// enumeration `for a in [0,d) for b in (a,d)` — the same order Phase 1
// fills planes in.
[[gnu::always_inline]] inline std::size_t
plane_idx(std::size_t d, std::size_t a, std::size_t b) noexcept {
    std::size_t idx = 0;
    for (std::size_t k = 0; k < a; ++k) {
        idx += d - 1 - k;
    }
    return idx + (b - a - 1);
}

// Per-row forward flat offset to the +ĵ neighbour for directions ≥ 1 (dir 0 is
// per-x). Fills `off[j]` (magnitude) and `wrap[j]` (true → subtract, crossing the
// seam) from the row index r. Shared by the U(1) strided plane sweeps; the same
// odometer the SU(N) kernels use. Requires d ≤ 4.
[[gnu::always_inline]] inline void
row_fwd_offsets(std::size_t r, Indexing::SizeVec const& sh, std::size_t d,
                std::size_t const (&stg)[4], std::size_t (&off)[4], bool (&wrap)[4]) noexcept {
    std::size_t rr = r;
    for (std::size_t j = 1; j < d; ++j) {
        std::size_t const lj = sh[j];
        std::size_t const cj = rr % lj;
        rr /= lj;
        off[j]  = (cj + 1 < lj) ? stg[j] : (lj - 1) * stg[j];
        wrap[j] = (cj + 1 == lj);
    }
}

}  // namespace impl

template <>
struct wilson_kernels<math::group::U1> {
    using G = math::group::U1;

    // Chunk-alignment granularity for the parallel sin/cos transcendental passes:
    // ≥ any SIMD width, so a non-final chunk always takes the same batched Sleef
    // path as a serial full sweep (bit-identical). The link-gather (Phase 2) is
    // per-site and uses gran = 1. The chunk *size* is derived from bytes_per_site.
    static constexpr std::size_t k_simd_gran = 16;

    // Table-free plaquette angle fill over the row-aligned range [base, base+cnt)
    // of one (a, b) plane: sp[s] = θ_p(s) with the two forward neighbours (s+â,
    // s+b̂) computed from the lattice strides (row-nested sweep, dir 0 per-x).
    // Caller applies sin / sincos to the filled contiguous scratch. Requires d≤4.
    template <class T>
    static void plaq_fill_range_(T const* u_a,
                                 T const* u_b,
                                 double* sp,
                                 std::size_t a,
                                 std::size_t b,
                                 Indexing::SizeVec const& sh,
                                 std::size_t d,
                                 std::size_t l0,
                                 std::size_t const (&stg)[4],
                                 std::size_t base,
                                 std::size_t cnt) noexcept {
        std::size_t const r0    = base / l0;
        std::size_t const nrows = cnt / l0;
        for (std::size_t r = r0; r < r0 + nrows; ++r) {
            std::size_t const row = r * l0;
            std::size_t off[4]    = {0, 0, 0, 0};
            bool wr[4]            = {false, false, false, false};
            impl::row_fwd_offsets(r, sh, d, stg, off, wr);
            auto nx = [&](std::size_t s, std::size_t x, std::size_t j) -> std::size_t {
                if (j == 0) {
                    return (x + 1 == l0) ? s - (l0 - 1) : s + 1;
                }
                return wr[j] ? s - off[j] : s + off[j];
            };
            for (std::size_t x = 0; x < l0; ++x) {
                std::size_t const s    = row + x;
                std::size_t const s_pa = nx(s, x, a);
                std::size_t const s_pb = nx(s, x, b);
                sp[s] = static_cast<double>(u1_plaq(u_a[s], u_b[s_pa], u_a[s_pb], u_b[s]));
            }
        }
    }

    // Fill the persistent nplanes*ns sinP scratch: one Sleef-batched sin pass
    // per (mu, nu) plane (mu < nu), chunked so the fill itself worksplits.
    // Pure sin-only fill — used by compute_force/compute_force_and_kick; the
    // fused s_full_and_force path below has its own sincos+reduce fill.
    template <class T>
    static void fill_sin_planes_(MatrixLinkLattice<G, T> const& u,
                                 double* sinP,
                                 std::size_t d,
                                 std::size_t ns) noexcept {
        auto const& sh       = u.shape();
        std::size_t const l0 = sh[0];
        std::size_t stg[4]   = {1, 0, 0, 0};
        for (std::size_t j = 1; j < d; ++j) {
            stg[j] = stg[j - 1] * sh[j - 1];
        }
        std::size_t pidx = 0;
        for (std::size_t a = 0; a < d; ++a) {
            T const* const u_a = u.mu_block_data(a);
            for (std::size_t b = a + 1; b < d; ++b) {
                T const* const u_b = u.mu_block_data(b);
                double* const sp   = sinP + (pidx * ns);
                reticolo::exec::field_visit(
                    u, k_simd_gran, [&, u_a, u_b, sp](std::size_t base, std::size_t cnt) {
                        plaq_fill_range_<T>(u_a, u_b, sp, a, b, sh, d, l0, stg, base, cnt);
                        math::sin_batch(sp + base, sp + base, cnt);
                    });
                ++pidx;
            }
        }
    }

    // Phase 2 — link-centric gather over sites [base, base+cnt). Each (mu, s)
    // is written exactly once (no other chunk touches it), so this is safe to
    // worksplit over disjoint site ranges for any thread count.
    //
    //   F_mu(x) = scale * sum_{nu != mu} sgn(mu,nu) *
    //               [ sinP[plane(mu,nu)][x] - sinP[plane(mu,nu)][prev(x,nu)] ]
    template <bool Fused, class T>
    static void force_from_sinp_range_(double const* sinP,
                                       MatrixLinkLattice<G, T> const& u,
                                       MatrixLinkLattice<G, T>& out,
                                       double scale,
                                       std::size_t base,
                                       std::size_t cnt) noexcept {
        auto const& sh       = u.shape();
        std::size_t const d  = u.ndims();
        std::size_t const ns = u.nsites();
        std::size_t const l0 = sh[0];
        std::size_t stg[4]   = {1, 0, 0, 0};
        for (std::size_t j = 1; j < d; ++j) {
            stg[j] = stg[j - 1] * sh[j - 1];
        }
        std::size_t const r0    = base / l0;
        std::size_t const nrows = cnt / l0;
        for (std::size_t mu = 0; mu < d; ++mu) {
            T* const out_mu_blk = out.mu_block_data(mu);
            // Precompute, once per mu, each nu's (sinP plane pointer, sign,
            // direction). 4 is the cap — d ≤ 4 (enforced on the gauge field).
            double const* sp_nu[4];
            double sgn_nu[4];
            std::size_t dir_nu[4];
            std::size_t n_nu = 0;
            for (std::size_t nu = 0; nu < d; ++nu) {
                if (nu == mu) {
                    continue;
                }
                std::size_t const a = mu < nu ? mu : nu;
                std::size_t const b = mu < nu ? nu : mu;
                sp_nu[n_nu]         = sinP + (impl::plane_idx(d, a, b) * ns);
                sgn_nu[n_nu]        = (mu < nu) ? 1.0 : -1.0;
                dir_nu[n_nu]        = nu;
                ++n_nu;
            }
            // Row-nested strided gather: prev(s, ν) from the lattice strides.
            for (std::size_t r = r0; r < r0 + nrows; ++r) {
                std::size_t const row = r * l0;
                std::size_t bo[4]     = {0, 0, 0, 0};
                bool bw[4]            = {false, false, false, false};
                std::size_t rr        = r;
                for (std::size_t j = 1; j < d; ++j) {
                    std::size_t const lj = sh[j];
                    std::size_t const cj = rr % lj;
                    rr /= lj;
                    bo[j] = (cj > 0) ? stg[j] : (lj - 1) * stg[j];
                    bw[j] = (cj == 0);
                }
                auto pv = [&](std::size_t s, std::size_t x, std::size_t j) -> std::size_t {
                    if (j == 0) {
                        return (x == 0) ? s + (l0 - 1) : s - 1;
                    }
                    return bw[j] ? s + bo[j] : s - bo[j];
                };
                for (std::size_t x = 0; x < l0; ++x) {
                    std::size_t const s = row + x;
                    double acc          = 0.0;
                    for (std::size_t k = 0; k < n_nu; ++k) {
                        std::size_t const s_prev = pv(s, x, dir_nu[k]);
                        acc += sgn_nu[k] * (sp_nu[k][s] - sp_nu[k][s_prev]);
                    }
                    double const val = scale * acc;
                    if constexpr (Fused) {
                        out_mu_blk[s] += static_cast<T>(val);
                    } else {
                        out_mu_blk[s] = static_cast<T>(val);
                    }
                }
            }
        }
    }

    // Link-centric Wilson force: Phase 1 (fill) then Phase 2 (gather), both
    // chunked so the whole pass worksplits. Write-once → no zero-init.
    template <class T>
    static void compute_force(MatrixLinkLattice<G, T> const& u,
                              MatrixLinkLattice<G, T>& force,
                              double beta_over_n) noexcept {
        std::size_t const d       = u.ndims();
        std::size_t const ns      = u.nsites();
        std::size_t const nplanes = (d * (d - 1)) / 2;
        double* const sinP        = impl::plane_scratch(nplanes * ns);
        fill_sin_planes_(u, sinP, d, ns);
        reticolo::exec::field_visit(u, 1, [&](std::size_t base, std::size_t cnt) {
            force_from_sinp_range_<false>(sinP, u, force, -beta_over_n, base, cnt);
        });
    }

    // Fused force-and-kick: same link-centric gather, accumulated straight
    // into the momentum field instead of a dedicated force buffer.
    template <class T>
    static void compute_force_and_kick(MatrixLinkLattice<G, T> const& u,
                                       MatrixLinkLattice<G, T>& mom,
                                       double beta_over_n,
                                       double k_dt) noexcept {
        std::size_t const d       = u.ndims();
        std::size_t const ns      = u.nsites();
        std::size_t const nplanes = (d * (d - 1)) / 2;
        double* const sinP        = impl::plane_scratch(nplanes * ns);
        fill_sin_planes_(u, sinP, d, ns);
        double const scale = -k_dt * beta_over_n;
        reticolo::exec::field_visit(u, 1, [&](std::size_t base, std::size_t cnt) {
            force_from_sinp_range_<true>(sinP, u, mom, scale, base, cnt);
        });
    }

    // Sigma Re Tr U_p over sites [base, base+cnt) of one (mu, nu) plane — the
    // Wilson `s_full` fast path. Pure per-range worker (no threading here);
    // `Wilson<G>::s_full_uncached` worksplits by calling this over a fixed
    // block partition via `field_reduce`.
    template <class T>
    static double s_full_plane_range(MatrixLinkLattice<G, T> const& u,
                                     std::size_t mu,
                                     std::size_t nu,
                                     std::size_t base,
                                     std::size_t cnt) noexcept {
        auto const& sh          = u.shape();
        std::size_t const d     = u.ndims();
        std::size_t const l0    = sh[0];
        T const* const u_mu_blk = u.mu_block_data(mu);
        T const* const u_nu_blk = u.mu_block_data(nu);
        double* const buf       = impl::plane_scratch(cnt);
        std::size_t stg[4]      = {1, 0, 0, 0};
        for (std::size_t j = 1; j < d; ++j) {
            stg[j] = stg[j - 1] * sh[j - 1];
        }
        std::size_t const r0    = base / l0;
        std::size_t const nrows = cnt / l0;
        for (std::size_t r = r0; r < r0 + nrows; ++r) {
            std::size_t const row = r * l0;
            std::size_t off[4]    = {0, 0, 0, 0};
            bool wr[4]            = {false, false, false, false};
            impl::row_fwd_offsets(r, sh, d, stg, off, wr);
            auto nx = [&](std::size_t s, std::size_t x, std::size_t j) -> std::size_t {
                if (j == 0) {
                    return (x + 1 == l0) ? s - (l0 - 1) : s + 1;
                }
                return wr[j] ? s - off[j] : s + off[j];
            };
            for (std::size_t x = 0; x < l0; ++x) {
                std::size_t const s     = row + x;
                std::size_t const s_pmu = nx(s, x, mu);
                std::size_t const s_pnu = nx(s, x, nu);
                buf[s - base]           = static_cast<double>(
                    u1_plaq(u_mu_blk[s], u_nu_blk[s_pmu], u_mu_blk[s_pnu], u_nu_blk[s]));
            }
        }
        math::cos_batch(buf, buf, cnt);
        double accum = 0.0;
        {
#if defined(__clang__)
            _Pragma("clang fp reassociate(on)")
#endif
                for (std::size_t i = 0; i < cnt; ++i) {
                accum += buf[i];
            }
        }
        return accum;
    }

    // Whole-plane Sigma Re Tr U_p (serial). Thin wrapper — `Wilson<G>::s_full`
    // binds to `s_full_plane_range` first (worksplits); this stays for any
    // caller that wants the whole-plane sum directly.
    template <class T>
    static double s_full_plane_re_tr_sum(MatrixLinkLattice<G, T> const& u,
                                         std::size_t mu,
                                         std::size_t nu) noexcept {
        return s_full_plane_range<T>(u, mu, nu, 0, u.nsites());
    }

    // Fused Sigma Re Tr U_p + link-centric force in one pass: Phase 1 fills
    // sinP AND folds cos(theta_p) into `accum` via one sincos_batch per plane
    // (deterministic chunked reduce); Phase 2 is the same gather as
    // `compute_force`. Returns the plaquette Re-Tr sum (the caller applies
    // the beta*n_plaq combine).
    template <class T>
    static double s_full_and_force(MatrixLinkLattice<G, T>& force_of,
                                   MatrixLinkLattice<G, T> const& u,
                                   double beta_over_n) noexcept {
        std::size_t const d       = u.ndims();
        std::size_t const ns      = u.nsites();
        std::size_t const nplanes = (d * (d - 1)) / 2;
        double* const scratch     = impl::plane_scratch((nplanes + 1) * ns);
        double* const sinP        = scratch;
        double* const cbuf        = scratch + (nplanes * ns);
        auto const& sh            = u.shape();
        std::size_t const l0      = sh[0];
        std::size_t stg[4]        = {1, 0, 0, 0};
        for (std::size_t j = 1; j < d; ++j) {
            stg[j] = stg[j - 1] * sh[j - 1];
        }
        double accum     = 0.0;
        std::size_t pidx = 0;
        for (std::size_t a = 0; a < d; ++a) {
            T const* const u_a = u.mu_block_data(a);
            for (std::size_t b = a + 1; b < d; ++b) {
                T const* const u_b = u.mu_block_data(b);
                double* const sp   = sinP + (pidx * ns);
                accum += reticolo::exec::field_reduce(
                    u, k_simd_gran, [&, u_a, u_b, sp](std::size_t base, std::size_t cnt) {
                        std::size_t const end = base + cnt;
                        plaq_fill_range_<T>(u_a, u_b, cbuf, a, b, sh, d, l0, stg, base, cnt);
                        math::sincos_batch(sp + base, cbuf + base, cbuf + base, cnt);
                        double sm = 0.0;
                        {
#if defined(__clang__)
                            _Pragma("clang fp reassociate(on)")
#endif
                                for (std::size_t s = base; s < end; ++s) {
                                sm += cbuf[s];
                            }
                        }
                        return sm;
                    });
                ++pidx;
            }
        }
        reticolo::exec::field_visit(u, 1, [&](std::size_t base, std::size_t cnt) {
            force_from_sinp_range_<false>(sinP, u, force_of, -beta_over_n, base, cnt);
        });
        return accum;
    }
};

}  // namespace reticolo::action::formula
