#pragma once

#include <reticolo/core/link_lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <concepts>
#include <cstddef>

namespace reticolo::gauge {

// =============================================================================
//  Concept lattice for link-field (gauge) actions. Mirrors `action/concepts.hpp`
//  but on `LinkLattice<F>` and on a (Site, mu) elementary update.
//
//  Every concept takes `LinkLattice<F> const&`. Actions must not mutate the
//  field. Force kernels receive a separate `LinkLattice<F>&` to write into.
// =============================================================================

// Baseline: enough to drive a link Metropolis sweep.
template <class A, class F>
concept LinkLocalAction =
    requires(A const& a, LinkLattice<F> const& l, Site x, std::size_t mu, F nv) {
        { a.s_local(l, x, mu) } -> std::convertible_to<double>;
        { a.ds_local(l, x, mu, nv) } -> std::convertible_to<double>;
    };

// Refinement: full action total. Required for HMC dH evaluation.
template <class A, class F>
concept HasLinkSEff = LinkLocalAction<A, F> && requires(A const& a, LinkLattice<F> const& l) {
    { a.s_full(l) } -> std::convertible_to<double>;
};

// Refinement: MD force. The kernel writes -dS/dtheta into `force`.
template <class A, class F>
concept HasLinkForce =
    LinkLocalAction<A, F> && requires(A const& a, LinkLattice<F> const& l, LinkLattice<F>& force) {
        { a.compute_force(l, force) };
    };

// Refinement: fused force + kick. Computes F_mu(x) and applies
// mom_mu(x) += k * F_mu(x) in one pass without materialising a force lattice.
template <class A, class F>
concept HasLinkFusedKick =
    HasLinkForce<A, F> && requires(A const& a, LinkLattice<F> const& l, LinkLattice<F>& mom, F k) {
        { a.compute_force_and_kick(l, mom, k) };
    };

}  // namespace reticolo::gauge
