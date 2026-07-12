#pragma once

#include <reticolo/core/field/field_traits.hpp>

#include <concepts>

namespace reticolo::action {

// Concept lattice — HMC only.
//
// Every action is a plain struct. The HMC updater needs exactly two members:
// the full action total and the molecular-dynamics force. Both are
// field-agnostic — they take the action's `Field` (a `Lattice<F>` for site
// actions, a `LinkLattice<F>` / `MatrixLinkLattice<G,F>` for gauge actions)
// and never mention a site index — so one concept set covers scalar and gauge
// alike. (Earlier revisions carried a separate `gauge::` family solely because
// the Metropolis elementary update had a different arity, `(Site, mu)` vs
// `Site`. With Metropolis gone, that distinction is gone with it.)
//
// Everything beyond the baseline is an opt-in refinement: the action exposes
// one more member and a corresponding concept matches, selecting the fused
// kick or the complex-LLR constraint path at instantiation. No base class, no
// virtual dispatch, no macros.

// Baseline: enough to run HMC. `compute_force` OVERWRITES `force` with
// -dS/dfield — it fully produces the buffer and never reads its prior
// contents, so callers don't pre-zero. (Actions that accumulate plane-by-plane
// internally must zero on entry to honour this.)
template <class A, class Field>
concept HmcAction = requires(A const& a, Field const& l, Field& f) {
    { a.s_full(l) } -> std::convertible_to<double>;
    { a.compute_force(l, f) };
};

// Refinement: fused force + kick. `compute_force_and_kick(field, mom, k)`
// computes the per-site force F(x) and applies `mom(x) += k * F(x)` in a single
// pass — never materialising the force lattice. The HMC integrator prefers this
// path when present; actions that only implement `compute_force` keep working.
template <class A, class Field>
concept HasFusedKick =
    HmcAction<A, Field> &&
    requires(A const& a, Field const& l, Field& mom, real_scalar_t<typename Field::value_type> k) {
        { a.compute_force_and_kick(l, mom, k) };
    };

// Refinement: action has a complex-valued total, decomposed as S = S_R + i*S_I.
// `s_full` returns the real (phase-quenched) part driving HMC and `s_imag`
// returns the imaginary part — typically the LLR constraint observable when the
// full theory has a sign problem. `compute_force_imag` writes the gradient of
// `s_imag` into a separate buffer; the LLR window combines them with the right
// coefficients.
template <class A, class Field>
concept HasImagPart = HmcAction<A, Field> && requires(A const& a, Field const& l, Field& force) {
    { a.s_imag(l) } -> std::convertible_to<double>;
    { a.compute_force_imag(l, force) };
};

}  // namespace reticolo::action
