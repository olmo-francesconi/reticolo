#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <concepts>
#include <cstddef>

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

// Refinement: action provides its own Metropolis proposal kernel. Selected at
// updater instantiation via `if constexpr`; the fallback is a Gaussian proposal.
template <class A, class F, class R>
concept HasProposal =
    LocalAction<A, F> && Rng<R> && requires(A const& a, Lattice<F> const& l, Site x, R& rng) {
        { a.propose(l, x, rng) } -> std::convertible_to<F>;
    };

// Refinement: action admits a Wolff cluster embedding. The cluster updater
// is action-agnostic; it asks the action for an axis, a reflection, and a
// link-acceptance probability.
template <class A, class F, class R>
concept WolffEmbeddable = LocalAction<A, F> && Rng<R> && requires(A const& a, F v, R& rng) {
    typename A::axis_type;
    { a.wolff_random_axis(rng) } -> std::same_as<typename A::axis_type>;
    { a.wolff_reflect(v, typename A::axis_type{}) } -> std::same_as<F>;
    { a.wolff_beta() } -> std::convertible_to<double>;
};

}  // namespace reticolo::action
