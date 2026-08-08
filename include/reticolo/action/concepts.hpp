#pragma once

#include <reticolo/core/field/field_traits.hpp>
#include <reticolo/core/field/lattice.hpp>

#include <concepts>
#include <cstddef>

namespace reticolo::action {

// Concept lattice — the global (HMC) baseline plus the local-update refinements.
//
// Every action is a plain struct. The HMC updater needs exactly two members:
// the full action total and the molecular-dynamics force. Both are
// field-agnostic — they take the action's `Field` (a `Lattice<F>` for site
// actions, a `LinkLattice<F>` / `MatrixLinkLattice<G,F>` for gauge actions)
// and never mention a site index — so one concept set covers scalar and gauge
// alike.
//
// A LOCAL updater (updater::Metropolis) needs one thing more: the change in S
// under a single-element move. `LocalAction` below covers site, complex and link
// fields with ONE signature, by keeping the elementary move — and therefore its
// arity — entirely inside the action.
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

// What one colour of a local sweep did. Returned by `metropolis_stencil` below —
// the aggregate an updater needs and the only thing it gets back.
struct LocalStats {
    std::size_t n_accepted = 0;
    std::size_t n_attempts = 0;
    double ds              = 0.0;  // Σ of the ACCEPTED ΔS — carries the running S

    LocalStats& operator+=(LocalStats const& o) noexcept {
        n_accepted += o.n_accepted;
        n_attempts += o.n_attempts;
        ds += o.ds;
        return *this;
    }

    // The blessed cross-updater accessor (see updater::Updater).
    [[nodiscard]] double acceptance() const noexcept {
        return n_attempts == 0 ? 0.0
                               : static_cast<double>(n_accepted) / static_cast<double>(n_attempts);
    }
};

// Refinement for LOCAL updates (updater::Metropolis). It is the exact peer of
// `HasFusedKick`: a lattice-level operation the action family drives, taking
// buffers and a scalar and returning an aggregate. The action owns the traversal
// and the local-energy evaluation — it is the only layer that knows how its
// kernels sweep a lattice — and the updater owns the algorithm and the
// randomness. No kernel, no site index and no callback crosses the boundary,
// exactly as with `compute_force(l, force)` and `compute_force_and_kick(l, mom, k)`.
//
//   n_colors(l)                                    -> int
//   metropolis_stencil(l, color, noise, logu, sigma) -> LocalStats
//
// A sweep is one pass per colour. `metropolis_stencil` updates every element of the
// given colour in place:
//     prop = move(element, sigma · noise)      accept iff  −ΔS ≥ logu
//
// `noise` is a SIBLING of the field — so a link field gets algebra-valued noise
// per link, a complex field an independent complex Gaussian — and `logu` is a
// REAL, SITE-shaped field of accept thresholds. Both are filled by the UPDATER.
// Passing pre-drawn randomness as plain fields is what keeps the action
// physics-only and randomness-free, and keeps the per-slab stream binding (see
// updater::fill_gaussian) an updater concern.
//
// The accept test is `−ΔS ≥ log u` rather than `u < exp(−ΔS)`: the same
// condition, but the uniform is consumed unconditionally, so the draw count is a
// pure function of the lattice shape rather than of the chain's history — the
// property the slab↔stream binding rests on — and the transcendental batches.
//
// COLOURS ARE OPAQUE to the updater, which only loops `[0, n_colors(l))`. A site
// field has 2, the checkerboard parities. A LINK field has 2·ndims, because its
// elementary move is per link and the safe colouring is (direction, site parity):
// a staple of (x,mu) only ever touches links of a different direction or of the
// opposite parity. Keeping the colour opaque is what lets ONE signature cover
// both — the elementary move's arity never crosses the interface, which is
// precisely the `(Site, mu)` vs `Site` split that forced two concept families in
// earlier revisions.
//
// It also settles the threshold buffer: `logu` carries one real per SITE, so a
// link sweep would otherwise reuse a site's threshold across its d directions —
// correlating accept decisions that must be independent. The updater therefore
// refills `logu` before EVERY colour, and fills `noise` once, since every element
// of `noise` is consumed exactly once across a whole sweep in either family.
template <class A, class Field>
concept LocalAction =
    HmcAction<A, Field> && requires(A const& a,
                                    Field& l,
                                    Field const& noise,
                                    Lattice<real_scalar_t<typename Field::value_type>> const& logu,
                                    int color,
                                    real_scalar_t<typename Field::value_type> sigma) {
        { a.n_colors(l) } -> std::convertible_to<int>;
        { a.metropolis_stencil(l, color, noise, logu, sigma) } -> std::convertible_to<LocalStats>;
    };

}  // namespace reticolo::action
