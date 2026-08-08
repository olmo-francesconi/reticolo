#pragma once

#include <reticolo/action/concepts.hpp>
#include <reticolo/action/formula/window_formula.hpp>
#include <reticolo/core/exec/field_ops.hpp>
#include <reticolo/core/exec/nn_checkerboard.hpp>
#include <reticolo/core/field/field_traits.hpp>
#include <reticolo/core/field/lattice.hpp>

#include <cstddef>
#include <optional>
#include <string_view>
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
// byte-for-byte (same formula calls, same caches); an arbitrary observable runs
// the generic two-pass force `F_base + scale·F_Q`, collapsing Q's value and
// gradient into ONE sweep when the observable exposes a fused `s_full_and_force`.

// --- constraint policies ------------------------------------------------------

// `k_name` is the on-disk tag for the window mode; analysis code needs it to
// know which reconstruction applies. Mode A (self) tilts by (1+a)·S, so the LLR
// fixed point is a = dln(rho)/dS − 1 and the DoS slope is (1+a); every other
// constraint tilts by a·Q, so the slope is a. Getting this wrong shifts ln(rho)
// by a term linear in the constrained variable.

// Q = the base action (window on S itself). A tag: the force fuses.
struct SelfConstraint {
    static constexpr bool k_self             = true;
    static constexpr std::string_view k_name = "self";
};

// Q = the base's imaginary observable S_I (sign-problem LLR). Delegates to the
// base's `HasImagPart` surface; its cache is the base's `s_imag` cache.
struct ImagConstraint {
    static constexpr bool k_self             = false;
    static constexpr std::string_view k_name = "imag";

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
    static constexpr bool k_self             = false;
    static constexpr std::string_view k_name = "observable";
    Obs obs;

    template <class Base, class Field>
    [[nodiscard]] auto value(Base const& /*b*/, Field const& l) const noexcept {
        return obs.s_full(l);
    }
    template <class Base, class Field>
    void compute_force(Base const& /*b*/, Field const& l, Field& out) const noexcept {
        obs.compute_force(l, out);
    }
    // Fused Q + (−dQ/dfield) in one neighbour pass, present only when the
    // observable offers the fast-path (any `NNAction` leaf that declares
    // `s_full_and_force`, e.g. Phi4, or `Wilson<G>`). Halves the sweeps per MD
    // force evaluation; like every `s_full_and_force`, it leaves the
    // observable's `last_s_full` cache alone — HMC's h0/h1 `s_full` calls own it.
    template <class Base, class Field>
    [[nodiscard]] auto value_and_force(Base const& /*b*/, Field const& l, Field& out) const noexcept
        requires requires(Obs const& o, Field const& cl, Field& f) { o.s_full_and_force(cl, f); }
    {
        return obs.s_full_and_force(l, out);
    }
    template <class Base>
    [[nodiscard]] double last(Base const& /*b*/) const noexcept {
        return obs.last_s_full();
    }
    template <class Base>
    void restore(Base const& /*b*/, double v) const noexcept {
        obs.restore_last_s_full(v);
    }
    // The observable's per-site local energy, for the sequential windowed sweep.
    // Present only when the observable is a family that offers one (any NNAction
    // leaf with an `action_kernel`), so a windowed action over an observable that
    // has no local form simply fails `action::LocalAction` instead of failing to
    // compile inside the updater.
    template <class Base>
    [[nodiscard]] auto local_energy_kernel(Base const& /*b*/) const noexcept
        requires requires(Obs const& o) { o.local_energy_kernel(); }
    {
        return obs.local_energy_kernel();
    }
    template <class Base>
    [[nodiscard]] auto local_energy_scale(Base const& /*b*/) const noexcept
        requires requires(Obs const& o) { o.local_energy_scale(); }
    {
        return obs.local_energy_scale();
    }
};

// The default constraint for a base: imaginary part if it has a sign problem,
// otherwise the action itself — reproducing the previous auto-selected modes.
template <class Base, class Field>
using default_constraint_t =
    std::conditional_t<action::HasImagPart<Base, Field>, ImagConstraint, SelfConstraint>;

// Same parameter structure as every other action-carrying template here: the
// element and field types are not parameters, because `Base` names both.
template <class Action, class Constraint = void>
struct WindowedAction {
    using base_type       = Action;
    using value_type      = Action::value_type;
    using field_type      = Action::field_type;
    using scalar_t        = scalar_of_t<value_type>;
    using constraint_type = std::conditional_t<std::is_void_v<Constraint>,
                                               default_constraint_t<Action, field_type>,
                                               Constraint>;

    // Owned by value (not reference): each Replica carries its own base +
    // constraint so any mutable scratch/cache stays per-replica (OpenMP over
    // replicas).
    Action base;
    scalar_t a = scalar_t{0};
    // NOLINTNEXTLINE(readability-identifier-naming) physics convention E_n = window centre
    scalar_t E_n   = scalar_t{0};
    scalar_t delta = scalar_t{1};
    constraint_type constraint{};

    static constexpr bool k_self    = constraint_type::k_self;
    static constexpr bool k_complex = !k_self;  // there is a separate constraint observable

    // The current constraint value Q(field) (fresh sweep).
    [[nodiscard]] scalar_t constraint_value(field_type const& l) const noexcept {
        if constexpr (k_self) {
            return static_cast<scalar_t>(base.s_full(l));
        } else {
            return static_cast<scalar_t>(constraint.value(base, l));
        }
    }

    [[nodiscard]] scalar_t s_full(field_type const& l) const noexcept {
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

    void compute_force(field_type const& l, field_type& force) const noexcept {
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
            exec::scale_field(force, scale);
        } else {
            // Generic: F = F_base + (a + (Q − E_n)/δ²)·F_Q.
            base.compute_force(l, force);
            field_type& q_force  = scratch_(force.indexing());
            scalar_t const q     = constraint_value_and_force_(l, q_force);
            scalar_t const scale = formula::force_scale_imag(q, a, E_n, delta);
            exec::kick_add(force, q_force, scale);  // force += scale·F_Q
        }
    }

    void compute_force_and_kick(field_type const& l, field_type& mom, scalar_t k_dt) const noexcept
        requires action::HasFusedKick<Action, field_type>
    {
        if constexpr (k_self) {
            if constexpr (requires(field_type& f) { base.s_full_and_force(l, f); }) {
                field_type& f        = scratch_(mom.indexing());
                auto const s         = static_cast<scalar_t>(base.s_full_and_force(l, f));
                scalar_t const scale = formula::force_scale(s, a, E_n, delta);
                exec::kick_add(mom, f, k_dt * scale);
            } else {
                auto const s         = static_cast<scalar_t>(base.s_full(l));
                scalar_t const scale = formula::force_scale(s, a, E_n, delta);
                base.compute_force_and_kick(l, mom, k_dt * scale);
            }
        } else {
            // Fused combined kernel (F_R + scale·F_Q in one pass) when the
            // constraint offers it (ImagConstraint over a base that fuses);
            // else the two-pass form, which still fuses Q's own value+gradient
            // when the constraint exposes `value_and_force`.
            if constexpr (requires {
                              constraint.combined_and_kick(
                                  base, l, mom, scalar_t{1}, scalar_t{1}, k_dt);
                          }) {
                auto const q         = static_cast<scalar_t>(constraint.value(base, l));
                scalar_t const scale = formula::force_scale_imag(q, a, E_n, delta);
                constraint.combined_and_kick(base, l, mom, scalar_t{1}, scale, k_dt);
            } else {
                // Q's pass runs first (it only reads `l`, which no kick touches),
                // so the two `kick_add`s stay in base-then-Q order regardless.
                field_type& q_force  = scratch_(mom.indexing());
                scalar_t const q     = constraint_value_and_force_(l, q_force);
                scalar_t const scale = formula::force_scale_imag(q, a, E_n, delta);
                base.compute_force_and_kick(l, mom, k_dt);
                exec::kick_add(mom, q_force, k_dt * scale);
            }
        }
    }

    // --- local (Metropolis) update -------------------------------------------
    //
    // ONE colour, and a strictly sequential pass — not the two-colour parallel
    // checkerboard every other action uses. The reason is the window, and only
    // the window: `a·Q` is LINEAR in Q, so its per-site increment `a·ΔQ` is
    // purely local, but the Gaussian penalty is quadratic in a GLOBAL scalar, so
    // a single move contributes `[2(Q−E_n)ΔQ + ΔQ²]/2δ²` — a term that reads the
    // CURRENT Q. Accepting a move at one site therefore changes ΔS at every other
    // site of the same colour, which is exactly the independence the checkerboard
    // rests on. Freezing Q across a colour would restore the parallelism and break
    // detailed balance; carrying Q as a running scalar through a fixed site order
    // keeps detailed balance exactly (each single-site update is a proper
    // Metropolis step against the current configuration).
    //
    // The parallelism this gives up is parallelism the consumer never had: an LLR
    // replica runs its lattice passes serially inside the replica-parallel team.
    //
    // Available only when the base — and the constraint observable, when there is
    // a separate one — expose a local energy. `ImagConstraint` does not yet, so a
    // sign-problem windowed action simply fails `action::LocalAction` and stays on
    // HMC rather than silently sampling something else.
    static constexpr bool k_local = requires(Action const& b) { b.local_energy_kernel(); } &&
                                    (k_self || requires(constraint_type const& c, Action const& b) {
                                        c.local_energy_kernel(b);
                                        c.local_energy_scale(b);
                                    });

    [[nodiscard]] static constexpr int n_colors(field_type const& /*l*/) noexcept { return 1; }

    [[nodiscard]] LocalStats metropolis_stencil(field_type& l,
                                                int /*color*/,
                                                field_type const& noise,
                                                Lattice<scalar_t> const& logu,
                                                scalar_t sigma) const noexcept
        requires k_local
    {
        auto const kb   = base.local_energy_kernel();
        auto const sc_b = base.local_energy_scale();
        auto const kq   = constraint_local_kernel_();
        auto const sc_q = constraint_local_scale_();

        // Re-derive both running scalars instead of trusting the caches. A stale Q
        // here does not merely mis-report the bookkeeping — it enters the window
        // term and biases EVERY accept test in the sweep. And the field is
        // routinely mutated between sweeps by code that never passed through this
        // action: an LLR app hot-starts it after construction, and replica
        // exchange swaps whole configurations. One extra reduce, against a sweep
        // that already gathers 2·d neighbours per site, buys immunity to a class
        // of bug that would only surface deep in a production run.
        auto s_base = static_cast<scalar_t>(base.s_full(l));
        scalar_t q  = s_base;
        if constexpr (!k_self) {
            q = static_cast<scalar_t>(constraint.value(base, l));
        }

        value_type* const data     = l.data();
        value_type const* const nz = noise.data();
        scalar_t const* const lu   = logu.data();

        auto const stats = exec::nn_sequential_reduce<LocalStats>(
            l, [&](std::size_t i, value_type self, auto const& gather) {
                value_type const prop = self + (sigma * nz[i]);
                auto const ds_base =
                    static_cast<scalar_t>(sc_b * (kb(prop, gather) - kb(self, gather)));
                scalar_t dq = ds_base;
                if constexpr (!k_self) {
                    dq = static_cast<scalar_t>(sc_q * (kq(prop, gather) - kq(self, gather)));
                }
                scalar_t const dw = formula::windowed_delta(ds_base, dq, q, a, E_n, delta);
                if (-static_cast<double>(dw) >= static_cast<double>(lu[i])) {
                    data[i] = prop;
                    s_base += ds_base;
                    q += dq;
                    return LocalStats{
                        .n_accepted = 1, .n_attempts = 1, .ds = static_cast<double>(dw)};
                }
                return LocalStats{.n_accepted = 0, .n_attempts = 1, .ds = 0.0};
            });

        // Publish what the sweep already knows exactly. These are the caches
        // orch::llr::Replica reads for ⟨dE⟩, exchange and reporting — the same
        // ones HMC leaves behind at the end of a trajectory.
        base.restore_last_s_full(static_cast<double>(s_base));
        if constexpr (!k_self) {
            constraint.restore(base, static_cast<double>(q));
        }
        return stats;
    }

    // Lazy scratch field for the non-self F_Q pass (and the self fused merge).
    // Allocated on first force call and reused; avoids the per-MD-step
    // malloc/free. `mutable` because the force methods are const.
    mutable std::optional<field_type> scratch_storage{};

private:
    // The constraint observable's local energy. For the self constraint there is
    // no separate observable — ΔQ IS ΔS_base and these are never called — so they
    // resolve to the base's own, which keeps the declaration well-formed without
    // inventing a placeholder type.
    [[nodiscard]] auto constraint_local_kernel_() const noexcept {
        if constexpr (k_self) {
            return base.local_energy_kernel();
        } else {
            return constraint.local_energy_kernel(base);
        }
    }
    [[nodiscard]] auto constraint_local_scale_() const noexcept {
        if constexpr (k_self) {
            return base.local_energy_scale();
        } else {
            return constraint.local_energy_scale(base);
        }
    }

    // Q and F_Q for the non-self path: one fused sweep when the constraint
    // offers `value_and_force`, else the value sweep + the force sweep.
    [[nodiscard]] scalar_t constraint_value_and_force_(field_type const& l,
                                                       field_type& q_force) const noexcept {
        if constexpr (requires { constraint.value_and_force(base, l, q_force); }) {
            return static_cast<scalar_t>(constraint.value_and_force(base, l, q_force));
        } else {
            auto const q = static_cast<scalar_t>(constraint.value(base, l));
            constraint.compute_force(base, l, q_force);
            return q;
        }
    }

    [[nodiscard]] field_type& scratch_(std::shared_ptr<Indexing const> idx) const noexcept {
        if (!scratch_storage) {
            scratch_storage.emplace(std::move(idx));
        }
        return *scratch_storage;
    }
};

}  // namespace reticolo::action
