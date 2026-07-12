#pragma once

#include <reticolo/action/gauge/formula/wilson_kernels.hpp>
#include <reticolo/action/gauge/gauge_action.hpp>
#include <reticolo/core/log/log.hpp>
#include <reticolo/core/field/matrix_link_lattice.hpp>
#include <reticolo/core/exec/parallel.hpp>
#include <reticolo/core/field/site.hpp>
#include <reticolo/math/group/base.hpp>

// The shipped per-group Wilson kernel specializations, so `Wilson<G>` for any
// built-in group resolves from a single `wilson.hpp` include. An external group
// defines its own `wilson_kernels<MyGroup>` and includes it alongside this.
#include <reticolo/action/gauge/formula/wilson_su2.hpp>
#include <reticolo/action/gauge/formula/wilson_su3.hpp>
#include <reticolo/action/gauge/formula/wilson_u1.hpp>

#include <cstddef>

namespace reticolo::action {

// Generic Wilson plaquette action for any SU(N) (and U(1)) gauge group
// conforming to the `math::group::GaugeGroup` concept.
//
//     S_W = (β/N) · Σ_x Σ_{μ<ν} ( N − Re Tr U_{μν}(x) )
//         = β · n_plaq − (β/N) · Σ_p Re Tr U_p
//
// The hot loops walk each plaquette plane (μ, ν) with a table-free row-nested
// strided sweep (neighbours from the lattice strides, dim-0 inner, wrap peeled).
// The per-plaquette physics — Re Tr U_p and the staple force scatter — is the
// action-specific `formula::wilson_kernels<G>` (in action/gauge/formula/wilson_<g>.hpp),
// which is built on the group model `G`'s core operations. `G` itself carries
// only the group constants (`n_color`, `name`) and the HMC algebra hooks; the
// Wilson force lives here in the action layer, not in the group math.
//
// At N=1 with U=exp(iθ) this reduces line-for-line to `CompactU1` — the
// bit-identity is the design point (validated in the physics suite).

template <math::group::GaugeGroup G, class T = double>
struct Wilson : GaugeAction<Wilson<G, T>> {
    using value_type = T;
    using group_type = G;
    using field_type = MatrixLinkLattice<G, T>;
    using kernels    = formula::wilson_kernels<G>;

    T beta = T{0};

    // Uniform announce hook (called by reticolo::log::act): concept name +
    // gauge group, then the coupling on its own indented line.
    void describe(log::Entry& e) const {
        e.line("Wilson<{}>", G::name);
        e.param("β={:.3f}", beta);
    }

    // Returns double regardless of the field precision T: the plaquette sum is
    // accumulated in double, and the final β·n_plaq − (β/N)·Σ combine — a
    // difference of two ~volume-sized quantities — is also done in double to
    // avoid catastrophic cancellation when T = float.
    // `s_full_uncached` — the base `GaugeAction::s_full` calls this and writes
    // the `last_s_full` cache. The LLR Replica reads that cache after a
    // trajectory instead of re-sweeping; HMC snapshots/restores it on rollback.
    [[nodiscard]] double s_full_uncached(field_type const& U) const noexcept {
        std::size_t const d      = U.ndims();
        std::size_t const ns     = U.nsites();
        std::size_t const n_plaq = (d * (d - 1) / 2) * ns;
        double accum_re_tr       = 0.0;
        // The parallel range worker: the gauge base worksplits each (μ,ν) plane
        // over the canonical field partition with deterministic per-item partials.
        // field_reduce derives the threshold/chunk from the gauge footprint;
        // gran = k_gauge_batch keeps the batched plane sum on batch boundaries.
        constexpr std::size_t gran = math::group::k_gauge_batch<T>;
        for (std::size_t mu = 0; mu < d; ++mu) {
            for (std::size_t nu = mu + 1; nu < d; ++nu) {
                accum_re_tr +=
                    reticolo::exec::field_reduce(U, gran, [&](std::size_t base, std::size_t cnt) {
                        return kernels::template s_full_plane_range<T>(U, mu, nu, base, cnt);
                    });
            }
        }
        auto const beta_d        = static_cast<double>(beta);
        double const beta_over_n = beta_d / static_cast<double>(G::n_color);
        return (beta_d * static_cast<double>(n_plaq)) - (beta_over_n * accum_re_tr);
    }

    // `force_into` (invoked by the base `compute_force`) hands the group's Wilson
    // kernels the β/N prefactor. U(1) uses the per-plaquette sin scatter that
    // matches CompactU1 bit-for-bit; SU(N) uses the link-centric staple + TA[U·V].
    void force_into(field_type const& U, field_type& force) const noexcept {
        auto const beta_over_n_dbl = static_cast<double>(beta / static_cast<T>(G::n_color));
        // Each group's compute_force owns its own threading — SU(N) worksplit the
        // staple kernel over write-disjoint chunks via field_visit; U(1) runs its
        // two-phase sinP fill+gather — so the action layer stays uniform with no
        // parallel-vs-serial branch of its own.
        kernels::compute_force(U, force, beta_over_n_dbl);
    }

    // Fused force-and-kick: scatter the per-plaquette force straight into the
    // momentum field, skipping the dedicated force buffer. Enabled when the
    // group's kernels provide a fused kernel (all three ship one).
    void compute_force_and_kick(field_type const& U, field_type& mom, T k_dt) const noexcept
        requires requires(field_type const& cu, field_type& m) {
            kernels::compute_force_and_kick(cu, m, double{}, double{});
        }
    {
        auto const beta_over_n_dbl = static_cast<double>(beta / static_cast<T>(G::n_color));
        // Same uniformity as force_into: the group's fused kernel owns its
        // threading, so the fused scatter is one call with no dispatch here.
        kernels::compute_force_and_kick(U, mom, beta_over_n_dbl, static_cast<double>(k_dt));
    }

    // Fused S_base + force in one pass (LLR fast-path). Present only when the
    // group's kernels provide it — U(1) does (one sincos plane pass); SU(N) fall
    // back to the two-pass s_full + compute_force. Does not touch the cache.
    // The LLR WindowedAction probes for this member and uses it when present.
    [[nodiscard]] double s_full_and_force(field_type const& U, field_type& force) const noexcept
        requires requires(field_type& f, field_type const& u) {
            kernels::s_full_and_force(f, u, double{});
        }
    {
        std::size_t const d          = U.ndims();
        std::size_t const ns         = U.nsites();
        std::size_t const n_plaq     = (d * (d - 1) / 2) * ns;
        auto const beta_d            = static_cast<double>(beta);
        double const beta_over_n_dbl = beta_d / static_cast<double>(G::n_color);
        double const accum           = kernels::s_full_and_force(force, U, beta_over_n_dbl);
        return (beta_d * static_cast<double>(n_plaq)) - (beta_over_n_dbl * accum);
    }
};

}  // namespace reticolo::action
