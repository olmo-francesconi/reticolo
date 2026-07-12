#pragma once

#include <reticolo/action/concepts.hpp>
#include <reticolo/action/formula/window_formula.hpp>
#include <reticolo/core/field/field_traits.hpp>
#include <reticolo/core/field/lattice.hpp>
#include <reticolo/updater/hmc/integ_ops.hpp>

#include <cstddef>
#include <optional>
#include <type_traits>

namespace reticolo::action {

// Window parameters (a, E_n, delta) stay real-valued even when the field is
// complex — `scalar_of_t` is the canonical `reticolo::real_scalar_t`.
template <class T>
using scalar_of_t = real_scalar_t<T>;

// Windowed action — a base action `S` plus a Gaussian-penalty window on some
// constraint observable `Q`:
//
//     S_win = S_base + a·Q + (Q − E_n)² / (2δ²)
//
// It is itself an `HmcAction` (HMC integrates it directly), so it lives in
// `action/` next to the concepts — LLR is one consumer (`orch::llr::Replica`),
// but a plain multicanonical/umbrella-sampling app could use it with no LLR at
// all. **What defines the window is a policy**, the `Constraint`:
//
//   * `SelfConstraint`  — Q = the base action itself. `S_win = (1+a)S + …`; the
//     force fuses to ONE scaled pass. The default for real actions (old mode A).
//   * `ImagConstraint`  — Q = the base's imaginary part (`s_imag`), for
//     sign-problem actions. The default when the base is `HasImagPart` (old mode B).
//   * `ObservableConstraint<Obs>` — Q = ANY observable (`obs.value(l)` +
//     `obs.compute_force(l, force)`): magnetization, plaquette, topological
//     charge, … "simulate this action, window on that quantity."
//
// The self and imag defaults reproduce the previous two hardcoded modes
// byte-for-byte (same formula calls, same caches); an arbitrary observable is a
// new opt-in that runs the generic two-pass force `F_base + scale·F_Q`.

// --- constraint policies ------------------------------------------------------

// Q = the base action (window on S itself). A tag: the force fuses.
struct SelfConstraint {
    static constexpr bool k_self = true;
};

// Q = the base's imaginary observable S_I (sign-problem LLR). Delegates to the
// base's `HasImagPart` surface; its cache is the base's `s_imag` cache.
struct ImagConstraint {
    static constexpr bool k_self = false;

    template <class Base, class Field>
    [[nodiscard]] auto value(Base const& b, Field const& l) const noexcept {
        return b.s_imag(l);
    }
    template <class Base, class Field>
    void compute_force(Base const& b, Field const& l, Field& out) const noexcept {
        b.compute_force_imag(l, out);
    }
    template <class Base, class Field, class S>
    void combined_and_kick(
        Base const& b, Field const& l, Field& mom, S sr, S si, S k_dt) const noexcept {
        b.compute_force_combined_and_kick(l, mom, sr, si, k_dt);
    }
    template <class Base>
    [[nodiscard]] double last(Base const& b) const noexcept {
        return b.last_s_imag();
    }
    template <class Base>
    void restore(Base const& b, double v) const noexcept {
        b.restore_last_s_imag(v);
    }
};

// Q = an arbitrary observable, itself an ACTION (any `HmcAction`): its `s_full`
// IS the observable value Q(field) and its `compute_force` writes −dQ/dfield.
// So the observable is defined the SAME way as any action — a leaf that supplies
// per-site formula kernels (usually an `NNAction`) — and its value/gradient run
// through the shared parallel sweep engine automatically; no hand-rolled loop.
// Its own `SFullCache` is the constraint-value cache HMC rolls back on reject.
template <class Obs>
struct ObservableConstraint {
    static constexpr bool k_self = false;
    Obs obs;

    template <class Base, class Field>
    [[nodiscard]] auto value(Base const& /*b*/, Field const& l) const noexcept {
        return obs.s_full(l);
    }
    template <class Base, class Field>
    void compute_force(Base const& /*b*/, Field const& l, Field& out) const noexcept {
        obs.compute_force(l, out);
    }
    template <class Base>
    [[nodiscard]] double last(Base const& /*b*/) const noexcept {
        return obs.last_s_full();
    }
    template <class Base>
    void restore(Base const& /*b*/, double v) const noexcept {
        obs.restore_last_s_full(v);
    }
};

// The default constraint for a base: imaginary part if it has a sign problem,
// otherwise the action itself — reproducing the previous auto-selected modes.
template <class Base, class Field>
using default_constraint_t =
    std::conditional_t<action::HasImagPart<Base, Field>, ImagConstraint, SelfConstraint>;

template <class Base, class T = Base::value_type, class Field = Lattice<T>, class Constraint = void>
struct WindowedAction {
    using value_type      = T;
    using field_type      = Field;
    using scalar_t        = scalar_of_t<T>;
    using constraint_type = std::
        conditional_t<std::is_void_v<Constraint>, default_constraint_t<Base, Field>, Constraint>;

    // Owned by value (not reference): each Replica carries its own base +
    // constraint so any mutable scratch/cache stays per-replica (OpenMP over
    // replicas).
    Base base;
    scalar_t a = scalar_t{0};
    // NOLINTNEXTLINE(readability-identifier-naming) physics convention E_n = window centre
    scalar_t E_n   = scalar_t{0};
    scalar_t delta = scalar_t{1};
    constraint_type constraint{};

    static constexpr bool k_self    = constraint_type::k_self;
    static constexpr bool k_complex = !k_self;  // there is a separate constraint observable

    // The current constraint value Q(field) (fresh sweep).
    [[nodiscard]] scalar_t constraint_value(Field const& l) const noexcept {
        if constexpr (k_self) {
            return static_cast<scalar_t>(base.s_full(l));
        } else {
            return static_cast<scalar_t>(constraint.value(base, l));
        }
    }

    [[nodiscard]] scalar_t s_full(Field const& l) const noexcept {
        if constexpr (k_self) {
            return formula::windowed_value(static_cast<scalar_t>(base.s_full(l)), a, E_n, delta);
        } else {
            auto const q = static_cast<scalar_t>(constraint.value(base, l));
            return formula::windowed_value_complex(
                static_cast<scalar_t>(base.s_full(l)), q, a, E_n, delta);
        }
    }

    // ---- caches -------------------------------------------------------------
    // last_s_full is the base action cache (the sampled S). last_s_imag is the
    // CONSTRAINT-value cache (base's s_imag for ImagConstraint, the observable's
    // own for a custom one) — HMC's HasSImagCache rolls it back on reject, and
    // Replica reads it as the exchange/adaptation observable. `last_constraint`
    // is the unified accessor over both.
    [[nodiscard]] scalar_t last_s_full() const noexcept { return base.last_s_full(); }
    void restore_last_s_full(scalar_t v) const noexcept { base.restore_last_s_full(v); }

    [[nodiscard]] scalar_t last_s_imag() const noexcept
        requires(!constraint_type::k_self)
    {
        return static_cast<scalar_t>(constraint.last(base));
    }
    void restore_last_s_imag(scalar_t v) const noexcept
        requires(!constraint_type::k_self)
    {
        constraint.restore(base, static_cast<double>(v));
    }

    [[nodiscard]] scalar_t last_constraint() const noexcept {
        if constexpr (k_self) {
            return base.last_s_full();
        } else {
            return static_cast<scalar_t>(constraint.last(base));
        }
    }

    void compute_force(Field const& l, Field& force) const noexcept {
        if constexpr (k_self) {
            // Fused base kernel when available: one neighbour pass yields both
            // S_base and the force, dropping the separate full-lattice s_full.
            scalar_t s{};
            if constexpr (requires { base.s_full_and_force(l, force); }) {
                s = static_cast<scalar_t>(base.s_full_and_force(l, force));
            } else {
                base.compute_force(l, force);
                s = static_cast<scalar_t>(base.s_full(l));
            }
            scalar_t const scale = formula::force_scale(s, a, E_n, delta);
            T* const fp          = force.data();
            std::size_t const n  = flat_size(force);
            for (std::size_t i = 0; i < n; ++i) {
                fp[i] *= scale;
            }
        } else {
            // Generic: F = F_base + (a + (Q − E_n)/δ²)·F_Q.
            base.compute_force(l, force);
            auto const q         = static_cast<scalar_t>(constraint.value(base, l));
            scalar_t const scale = formula::force_scale_imag(q, a, E_n, delta);
            Field& q_force       = scratch_(force.indexing());
            constraint.compute_force(base, l, q_force);
            updater::integ::kick_add(force, q_force, scale);  // force += scale·F_Q
        }
    }

    void compute_force_and_kick(Field const& l, Field& mom, scalar_t k_dt) const noexcept
        requires action::HasFusedKick<Base, Field>
    {
        if constexpr (k_self) {
            if constexpr (requires(Field& f) { base.s_full_and_force(l, f); }) {
                Field& f             = scratch_(mom.indexing());
                auto const s         = static_cast<scalar_t>(base.s_full_and_force(l, f));
                scalar_t const scale = formula::force_scale(s, a, E_n, delta);
                updater::integ::kick_add(mom, f, k_dt * scale);
            } else {
                auto const s         = static_cast<scalar_t>(base.s_full(l));
                scalar_t const scale = formula::force_scale(s, a, E_n, delta);
                base.compute_force_and_kick(l, mom, k_dt * scale);
            }
        } else {
            auto const q         = static_cast<scalar_t>(constraint.value(base, l));
            scalar_t const scale = formula::force_scale_imag(q, a, E_n, delta);
            // Fused combined kernel (F_R + scale·F_Q in one pass) when the
            // constraint offers it (ImagConstraint over a base that fuses);
            // else the two-pass form.
            if constexpr (requires {
                              constraint.combined_and_kick(base, l, mom, scalar_t{1}, scale, k_dt);
                          }) {
                constraint.combined_and_kick(base, l, mom, scalar_t{1}, scale, k_dt);
            } else {
                base.compute_force_and_kick(l, mom, k_dt);
                Field& q_force = scratch_(mom.indexing());
                constraint.compute_force(base, l, q_force);
                updater::integ::kick_add(mom, q_force, k_dt * scale);
            }
        }
    }

    // Lazy scratch field for the non-self F_Q pass (and the self fused merge).
    // Allocated on first force call and reused; avoids the per-MD-step
    // malloc/free. `mutable` because the force methods are const.
    mutable std::optional<Field> scratch_storage{};

private:
    [[nodiscard]] Field& scratch_(std::shared_ptr<Indexing const> idx) const noexcept {
        if (!scratch_storage) {
            scratch_storage.emplace(std::move(idx));
        }
        return *scratch_storage;
    }
};

}  // namespace reticolo::action
