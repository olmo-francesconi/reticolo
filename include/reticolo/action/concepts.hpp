#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <concepts>

namespace reticolo::action {

// =============================================================================
//  Concept lattice.
//
//  Every action is a plain struct. The minimum interface (`LocalAction`) is
//  the two members an updater needs to do a Metropolis sweep: the local
//  contribution to S at a site, and the change in that contribution under a
//  proposed value. Everything else is an *opt-in refinement*: the action
//  exposes one additional member function and a corresponding concept matches.
//  This is how HMC, full-S diagnostics, and Wolff cluster moves are selected
//  at instantiation without growing `LocalAction` itself.
//
//  Every concept takes `Lattice<F> const&`. Actions must not mutate the field.
//  Force kernels receive a separate `Lattice<F>&` to write into.
//
//  No preprocessor inside any concept body. If/when parallelism returns,
//  it lands as a sibling concept (`HasForceInParallel`) and a separate updater
//  specialisation, never as a macro flip.
// =============================================================================

// Baseline: enough to drive a Metropolis sweep with a uniform / external proposal.
template <class A, class F>
concept LocalAction = requires(A const& a, Lattice<F> const& l, Site x, F nv) {
    { a.s_local(l, x) } -> std::convertible_to<double>;
    { a.ds_local(l, x, nv) } -> std::convertible_to<double>;
};

// Refinement: full action total. Required for HMC dH evaluation and for
// diagnostics that want S itself (not just local contributions).
template <class A, class F>
concept HasSEff = LocalAction<A, F> && requires(A const& a, Lattice<F> const& l) {
    { a.s_full(l) } -> std::convertible_to<double>;
};

// Refinement: molecular-dynamics force. The kernel writes -dS/dphi into `force`.
template <class A, class F>
concept HasForce =
    LocalAction<A, F> && requires(A const& a, Lattice<F> const& l, Lattice<F>& force) {
        { a.compute_force(l, force) };
    };

// Refinement: fused force + kick. `compute_force_and_kick(field, mom, k)` computes
// the per-site force F(x) and applies `mom(x) += k * F(x)` in a single pass — never
// materialising the force lattice in memory. The HMC integrator prefers this path
// when present; actions that only implement `compute_force` keep working unchanged.
template <class A, class F>
concept HasFusedKick =
    HasForce<A, F> && requires(A const& a, Lattice<F> const& l, Lattice<F>& mom, F k) {
        { a.compute_force_and_kick(l, mom, k) };
    };

// Refinement: action exposes ds_local as pure math on (phi, new_v, nbrs), where
// nbrs is the unweighted sum of all 2*ndims nearest neighbours of the site. The
// Metropolis sweep uses this path with `visit_nn` (direct-stride neighbour reads,
// vectorisation-friendly) when supported; otherwise it falls back to `ds_local`.
template <class A, class F>
concept HasDsLocalFromNbrs =
    LocalAction<A, F> && requires(A const& a, F phi, F new_v, F nbrs) {
        { a.ds_local_from_nbrs(phi, new_v, nbrs) } -> std::convertible_to<double>;
    };

// Refinement: action provides its own Metropolis proposal kernel. Selected at
// updater instantiation via `if constexpr`; the fallback is a Gaussian proposal.
template <class A, class F, class R>
concept HasProposal =
    LocalAction<A, F> && Rng<R> && requires(A const& a, Lattice<F> const& l, Site x, R& rng) {
        { a.propose(l, x, rng) } -> std::convertible_to<F>;
    };

// Refinement: action admits a Wolff cluster embedding. The cluster updater
// is action-agnostic; it asks the action for an axis, a reflection, and a
// link-acceptance probability. `wolff_link_p` returns the bond-activation
// probability between two ORIGINAL (pre-flip) field values across the given
// reflection — this keeps all action-specific physics (beta, coupling form,
// projection onto the reflection axis) inside the action itself.
template <class A, class F, class R>
concept WolffEmbeddable =
    LocalAction<A, F> && Rng<R> &&
    requires(A const& a, F const& v, typename A::axis_type const& axis, R& rng) {
        typename A::axis_type;
        { a.wolff_random_axis(rng) } -> std::same_as<typename A::axis_type>;
        { a.wolff_reflect(v, axis) } -> std::same_as<F>;
        { a.wolff_link_p(v, v, axis) } -> std::convertible_to<double>;
    };

}  // namespace reticolo::action
