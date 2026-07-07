#pragma once

#include <reticolo/action/gauge/detail/wilson_kernels.hpp>
#include <reticolo/action/gauge/formula/wilson_u1_formula.hpp>
#include <reticolo/core/indexing.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/parallel.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/math/gauge_group/u1.hpp>
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
// -sin P_{nu,mu}). Two-phase, both chunked via `parallel_map_ranges` /
// `parallel_reduce_ranges` so the whole pass worksplits:
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

namespace reticolo::action::detail {

namespace u1_detail {

// Thread-local scratch shared by every U(1) Wilson kernel entry point below.
// Each entry point calls this exactly once per invocation, with the total
// size it needs, and slices sub-spans out of the single pointer — so a
// resize inside one call can never invalidate a pointer live in another part
// of the same call.
[[gnu::always_inline]] inline double* plane_scratch(std::size_t n) noexcept {
    thread_local std::vector<double> buf;
    if (buf.size() < n) {
        buf.resize(n);
    }
    return buf.data();
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

}  // namespace u1_detail

template <>
struct wilson_kernels<gauge_group::U1> {
    using G = gauge_group::U1;

    // Chunk-alignment granularity for the parallel sin/cos transcendental passes:
    // ≥ any SIMD width, so a non-final chunk always takes the same batched Sleef
    // path as a serial full sweep (bit-identical). The link-gather (Phase 2) is
    // per-site and uses gran = 1. The chunk *size* is derived from bytes_per_site.
    static constexpr std::size_t k_simd_gran = 16;

    // Fill the persistent nplanes*ns sinP scratch: one Sleef-batched sin pass
    // per (mu, nu) plane (mu < nu), chunked so the fill itself worksplits.
    // Pure sin-only fill — used by compute_force/compute_force_and_kick; the
    // fused s_full_and_force path below has its own sincos+reduce fill.
    template <class T>
    static void fill_sin_planes_(MatrixLinkLattice<G, T> const& u,
                                 double* sinP,
                                 std::size_t d,
                                 std::size_t ns) noexcept {
        Indexing const& idx   = u.indexing_ref();
        std::size_t const bps = u.bytes_per_site();
        std::size_t pidx      = 0;
        for (std::size_t a = 0; a < d; ++a) {
            T const* const u_a = u.mu_block_data(a);
            for (std::size_t b = a + 1; b < d; ++b) {
                T const* const u_b = u.mu_block_data(b);
                double* const sp   = sinP + (pidx * ns);
                reticolo::detail::parallel_map_ranges(
                    ns, bps, k_simd_gran, [&, u_a, u_b, sp](std::size_t base, std::size_t cnt) {
                        std::size_t const end = base + cnt;
                        for (std::size_t s = base; s < end; ++s) {
                            std::size_t const s_pa = idx.next(Site{s}, a).value();
                            std::size_t const s_pb = idx.next(Site{s}, b).value();
                            sp[s] =
                                static_cast<double>(u1_plaq(u_a[s], u_b[s_pa], u_a[s_pb], u_b[s]));
                        }
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
        std::size_t const d   = u.ndims();
        std::size_t const ns  = u.nsites();
        Indexing const& idx   = u.indexing_ref();
        std::size_t const end = base + cnt;
        for (std::size_t mu = 0; mu < d; ++mu) {
            T* const out_mu_blk = out.mu_block_data(mu);
            // Precompute, once per mu, each nu's (sinP plane pointer, sign,
            // direction). 8 is a generous cap — no lattice-QFT theory this
            // tree targets exceeds 4 dimensions.
            double const* sp_nu[8];
            double sgn_nu[8];
            std::size_t dir_nu[8];
            std::size_t n_nu = 0;
            for (std::size_t nu = 0; nu < d; ++nu) {
                if (nu == mu) {
                    continue;
                }
                std::size_t const a = mu < nu ? mu : nu;
                std::size_t const b = mu < nu ? nu : mu;
                sp_nu[n_nu]         = sinP + (u1_detail::plane_idx(d, a, b) * ns);
                sgn_nu[n_nu]        = (mu < nu) ? 1.0 : -1.0;
                dir_nu[n_nu]        = nu;
                ++n_nu;
            }
            for (std::size_t s = base; s < end; ++s) {
                double acc = 0.0;
                for (std::size_t k = 0; k < n_nu; ++k) {
                    std::size_t const s_prev = idx.prev(Site{s}, dir_nu[k]).value();
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

    // Link-centric Wilson force: Phase 1 (fill) then Phase 2 (gather), both
    // chunked so the whole pass worksplits. Write-once → no zero-init.
    template <class T>
    static void compute_force(MatrixLinkLattice<G, T> const& u,
                              MatrixLinkLattice<G, T>& force,
                              double beta_over_n) noexcept {
        std::size_t const d       = u.ndims();
        std::size_t const ns      = u.nsites();
        std::size_t const nplanes = (d * (d - 1)) / 2;
        double* const sinP        = u1_detail::plane_scratch(nplanes * ns);
        fill_sin_planes_(u, sinP, d, ns);
        reticolo::detail::parallel_map_ranges(
            ns, u.bytes_per_site(), 1, [&](std::size_t base, std::size_t cnt) {
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
        double* const sinP        = u1_detail::plane_scratch(nplanes * ns);
        fill_sin_planes_(u, sinP, d, ns);
        double const scale = -k_dt * beta_over_n;
        reticolo::detail::parallel_map_ranges(
            ns, u.bytes_per_site(), 1, [&](std::size_t base, std::size_t cnt) {
                force_from_sinp_range_<true>(sinP, u, mom, scale, base, cnt);
            });
    }

    // Sigma Re Tr U_p over sites [base, base+cnt) of one (mu, nu) plane — the
    // Wilson `s_full` fast path. Pure per-range worker (no threading here);
    // `Wilson<G>::s_full_uncached` worksplits by calling this over a fixed
    // block partition via `parallel_reduce_ranges`.
    template <class T>
    static double s_full_plane_range(MatrixLinkLattice<G, T> const& u,
                                     std::size_t mu,
                                     std::size_t nu,
                                     std::size_t base,
                                     std::size_t cnt) noexcept {
        Indexing const& idx     = u.indexing_ref();
        T const* const u_mu_blk = u.mu_block_data(mu);
        T const* const u_nu_blk = u.mu_block_data(nu);
        double* const buf       = u1_detail::plane_scratch(cnt);
        std::size_t const end   = base + cnt;
        for (std::size_t s = base; s < end; ++s) {
            std::size_t const s_pmu = idx.next(Site{s}, mu).value();
            std::size_t const s_pnu = idx.next(Site{s}, nu).value();
            buf[s - base]           = static_cast<double>(
                u1_plaq(u_mu_blk[s], u_nu_blk[s_pmu], u_mu_blk[s_pnu], u_nu_blk[s]));
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
        double* const scratch     = u1_detail::plane_scratch((nplanes + 1) * ns);
        double* const sinP        = scratch;
        double* const cbuf        = scratch + (nplanes * ns);
        Indexing const& idx       = u.indexing_ref();
        std::size_t const bps     = u.bytes_per_site();
        double accum              = 0.0;
        std::size_t pidx          = 0;
        for (std::size_t a = 0; a < d; ++a) {
            T const* const u_a = u.mu_block_data(a);
            for (std::size_t b = a + 1; b < d; ++b) {
                T const* const u_b = u.mu_block_data(b);
                double* const sp   = sinP + (pidx * ns);
                accum += reticolo::detail::parallel_reduce_ranges(
                    ns, bps, k_simd_gran, [&, u_a, u_b, sp](std::size_t base, std::size_t cnt) {
                        std::size_t const end = base + cnt;
                        for (std::size_t s = base; s < end; ++s) {
                            std::size_t const s_pa = idx.next(Site{s}, a).value();
                            std::size_t const s_pb = idx.next(Site{s}, b).value();
                            cbuf[s] =
                                static_cast<double>(u1_plaq(u_a[s], u_b[s_pa], u_a[s_pb], u_b[s]));
                        }
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
        reticolo::detail::parallel_map_ranges(ns, bps, 1, [&](std::size_t base, std::size_t cnt) {
            force_from_sinp_range_<false>(sinP, u, force_of, -beta_over_n, base, cnt);
        });
        return accum;
    }
};

}  // namespace reticolo::action::detail
