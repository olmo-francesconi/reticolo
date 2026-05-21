#pragma once

#include <reticolo/action/detail/concepts.hpp>
#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/link_lattice.hpp>
#include <reticolo/core/site.hpp>

#include <complex>
#include <cstddef>
#include <optional>

namespace reticolo::llr {

// Strip std::complex<T> down to T; pass-through for non-complex types. Used so
// that LLR window parameters (a, E_n, delta) stay real-valued even when the
// field is complex.
template <class T>
struct scalar_of {
    using type = T;
};
template <class T>
struct scalar_of<std::complex<T>> {
    using type = T;
};
template <class T>
using scalar_of_t = typename scalar_of<T>::type;

// LLR-tilted action with Gaussian-penalty window. One template for both
// scalar (`Field = Lattice<T>`) and gauge (`Field = LinkLattice<T>`) base
// actions. Two modes picked at compile time from the base action's
// interface (only the scalar-field BoseGas hits mode B today):
//
// Mode A — real LLR, `Base` does NOT satisfy `HasImagPart`. The window
// constrains the base action itself:
//
//     S_LLR    = (1 + a) * S_base + (S_base - E_n)^2 / (2 * delta^2)
//     F_LLR(x) = (1 + a + (S_base - E_n) / delta^2) * F_base(x)
//
// Mode B — complex LLR, `Base` satisfies `HasImagPart`. HMC samples on the
// real (phase-quenched) part S_R and the window constrains the imaginary
// observable S_I = `base.s_imag(field)`:
//
//     S_LLR    = S_R + a * S_I + (S_I - E_n)^2 / (2 * delta^2)
//     F_LLR(x) = F_R(x) + (a + (S_I - E_n) / delta^2) * F_I(x)
//
// where F_R = base.compute_force, F_I = base.compute_force_imag. The NR/RM
// loop watches <S_I - E_n>; reconstructing ln rho(S_I) recovers the DoS of
// the imaginary part in the phase-quenched ensemble.
//
// `s_local` forwards to the base action; `ds_local` returns the full windowed
// delta. The Metropolis sweep maintains a running constraint value via the
// `begin_sweep` / `commit_accept` hooks (concept `HasSweepState` /
// `HasLinkSweepState`) so per-site delta math sees the right S_running
// without an O(V) recompute per site.
//
// Mode A delta: d((1+a)·S + (S-E_n)²/2δ²) = (1+a)·dS + ((S_run-E_n)·dS + dS²/2)/δ²
// Mode B delta: d(S_R + a·S_I + (S_I-E_n)²/2δ²)
//             = dS_R + a·dS_I + ((S_I_run-E_n)·dS_I + dS_I²/2)/δ²

template <class Base, class T = typename Base::value_type, class Field = Lattice<T>>
struct WindowedAction {
    using value_type = T;
    using field_type = Field;
    using scalar_t   = scalar_of_t<T>;

    // Owned by value (not reference): each Replica carries its own base so any
    // mutable scratch on the action stays per-replica. Required for OpenMP
    // parallelism over replicas.
    Base base;
    scalar_t a = scalar_t{0};
    // NOLINTNEXTLINE(readability-identifier-naming) physics convention E_n = window centre
    scalar_t E_n   = scalar_t{0};
    scalar_t delta = scalar_t{1};

    static constexpr bool k_complex = action::HasImagPart<Base, T>;

    // Running constraint value during a Metropolis sweep. Refreshed by
    // `begin_sweep` from the current field; incremented by `commit_accept`
    // on every accepted local move so subsequent `ds_local` calls in the
    // same sweep see the correct windowed delta. Mutable because the hooks
    // are logically const-action operations.
    mutable scalar_t s_constraint_running_{};

    // Site-arity (Field = Lattice<T>).
    [[nodiscard]] scalar_t s_local(Field const& l, Site x) const noexcept
        requires action::LocalAction<Base, T>
    {
        return base.s_local(l, x);
    }
    [[nodiscard]] scalar_t ds_local(Field const& l, Site x, T new_v) const noexcept
        requires action::LocalAction<Base, T>
    {
        scalar_t const ds_base = base.ds_local(l, x, new_v);
        scalar_t ds_constraint{};
        if constexpr (k_complex) {
            ds_constraint = base.ds_imag_local(l, x, new_v);
        } else {
            ds_constraint = ds_base;
        }
        scalar_t const inv2 = scalar_t{1} / (delta * delta);
        scalar_t const half = scalar_t{1} / scalar_t{2};
        scalar_t const window_ds =
            (((s_constraint_running_ - E_n) * ds_constraint) +
             (half * ds_constraint * ds_constraint)) *
            inv2;
        if constexpr (k_complex) {
            return ds_base + (a * ds_constraint) + window_ds;
        } else {
            return ((scalar_t{1} + a) * ds_base) + window_ds;
        }
    }

    void begin_sweep(Field const& l) const noexcept
        requires action::LocalAction<Base, T>
    {
        s_constraint_running_ = constraint_value(l);
    }

    void commit_accept(Field const& l, Site x, T new_v) const noexcept
        requires action::LocalAction<Base, T>
    {
        if constexpr (k_complex) {
            s_constraint_running_ += base.ds_imag_local(l, x, new_v);
        } else {
            s_constraint_running_ += base.ds_local(l, x, new_v);
        }
    }

    // (Site, mu)-arity (Field = LinkLattice<T>). Gauge LLR is mode-A only
    // today (no `HasImagPart` gauge action exists), so the constraint is the
    // base action itself and `ds_constraint == ds_base`.
    [[nodiscard]] scalar_t s_local(Field const& l, Site x, std::size_t mu) const noexcept
        requires gauge::LinkLocalAction<Base, T>
    {
        return base.s_local(l, x, mu);
    }
    [[nodiscard]] scalar_t ds_local(Field const& l, Site x, std::size_t mu, T new_v) const noexcept
        requires gauge::LinkLocalAction<Base, T>
    {
        scalar_t const ds_base = base.ds_local(l, x, mu, new_v);
        scalar_t const inv2    = scalar_t{1} / (delta * delta);
        scalar_t const half    = scalar_t{1} / scalar_t{2};
        scalar_t const window_ds =
            (((s_constraint_running_ - E_n) * ds_base) + (half * ds_base * ds_base)) *
            inv2;
        return ((scalar_t{1} + a) * ds_base) + window_ds;
    }

    void begin_sweep(Field const& l) const noexcept
        requires gauge::LinkLocalAction<Base, T>
    {
        s_constraint_running_ = constraint_value(l);
    }

    void commit_accept(Field const& l, Site x, std::size_t mu, T new_v) const noexcept
        requires gauge::LinkLocalAction<Base, T>
    {
        s_constraint_running_ += base.ds_local(l, x, mu, new_v);
    }

    [[nodiscard]] scalar_t constraint_value(Field const& l) const noexcept {
        if constexpr (k_complex) {
            return base.s_imag(l);
        } else {
            return base.s_full(l);
        }
    }

    [[nodiscard]] scalar_t s_full(Field const& l) const noexcept {
        scalar_t const q   = constraint_value(l);
        scalar_t const dq  = q - E_n;
        scalar_t const inv = scalar_t{1} / (scalar_t{2} * delta * delta);
        if constexpr (k_complex) {
            return base.s_full(l) + (a * q) + (dq * dq * inv);
        } else {
            return ((scalar_t{1} + a) * q) + (dq * dq * inv);
        }
    }

    // Raw-action cache pass-through to the base. The LLR Replica reads these
    // after a trajectory to skip the post-trajectory constraint sweep; HMC
    // snapshots and restores them across reject rollbacks. See
    // `HasCacheRollback` in `algorithm/hmc.hpp`.
    [[nodiscard]] scalar_t last_s_full() const noexcept { return base.last_s_full(); }
    void restore_last_s_full(scalar_t v) const noexcept { base.restore_last_s_full(v); }

    [[nodiscard]] scalar_t last_s_imag() const noexcept
        requires action::HasImagPart<Base, T>
    {
        return base.last_s_imag();
    }
    void restore_last_s_imag(scalar_t v) const noexcept
        requires action::HasImagPart<Base, T>
    {
        base.restore_last_s_imag(v);
    }

    void compute_force(Field const& l, Field& force) const noexcept {
        if constexpr (k_complex) {
            // Combined: F_R + (a + (S_I - E_n)/delta^2) * F_I.
            base.compute_force(l, force);
            scalar_t const s     = base.s_imag(l);
            scalar_t const scale = a + ((s - E_n) / (delta * delta));
            Field& imag_force    = imag_scratch_(force.indexing());
            base.compute_force_imag(l, imag_force);
            T* const fp         = force.data();
            T const* const ip   = imag_force.data();
            std::size_t const n = flat_size(force);
            for (std::size_t i = 0; i < n; ++i) {
                fp[i] += scale * ip[i];
            }
        } else {
            base.compute_force(l, force);
            scalar_t const s     = base.s_full(l);
            scalar_t const scale = scalar_t{1} + a + ((s - E_n) / (delta * delta));
            T* const fp          = force.data();
            std::size_t const n  = flat_size(force);
            for (std::size_t i = 0; i < n; ++i) {
                fp[i] *= scale;
            }
        }
    }

    void compute_force_and_kick(Field const& l, Field& mom, scalar_t k_dt) const noexcept
        requires action::HasFusedKick<Base, T> || gauge::HasLinkFusedKick<Base, T>
    {
        if constexpr (k_complex) {
            scalar_t const s     = base.s_imag(l);
            scalar_t const scale = a + ((s - E_n) / (delta * delta));
            // If the base action exposes a fused combined-force kernel
            // (F_R + scale·F_I in one pass directly into mom), use it —
            // skips the imag_force scratch buffer and the merge pass.
            // Otherwise fall back to the two-pass form.
            if constexpr (requires {
                              base.compute_force_combined_and_kick(
                                  l, mom, scalar_t{1}, scale, k_dt);
                          }) {
                base.compute_force_combined_and_kick(l, mom, scalar_t{1}, scale, k_dt);
            } else {
                base.compute_force_and_kick(l, mom, k_dt);
                Field& imag_force = imag_scratch_(mom.indexing());
                base.compute_force_imag(l, imag_force);
                T* const mp         = mom.data();
                T const* const ip   = imag_force.data();
                std::size_t const n = flat_size(mom);
                scalar_t const k    = k_dt * scale;
                for (std::size_t i = 0; i < n; ++i) {
                    mp[i] += k * ip[i];
                }
            }
        } else {
            scalar_t const s     = base.s_full(l);
            scalar_t const scale = scalar_t{1} + a + ((s - E_n) / (delta * delta));
            base.compute_force_and_kick(l, mom, k_dt * scale);
        }
    }

    // Lazy scratch buffer for the complex-LLR imag-force pass (mode B only).
    // Allocated on first force call and reused; avoids the per-MD-step
    // malloc/free that used to dominate the bose_gas_llr force path.
    // `mutable` because the force methods are const.
    mutable std::optional<Field> imag_scratch_storage{};

private:
    [[nodiscard]] Field& imag_scratch_(std::shared_ptr<Indexing const> idx) const noexcept {
        if (!imag_scratch_storage) {
            imag_scratch_storage.emplace(std::move(idx));
        }
        return *imag_scratch_storage;
    }
};

}  // namespace reticolo::llr
