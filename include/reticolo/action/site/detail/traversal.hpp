#pragma once

#include <reticolo/action/detail/traverse.hpp>
#include <reticolo/core/lattice.hpp>

#include <cstddef>
#include <utility>

// Site nearest-neighbour drivers — the identity-combine specialisation of the
// shared dimension-generic engine in <reticolo/action/detail/traverse.hpp>. A
// site action consumes the raw neighbour sum (self + Σ neighbours), so the
// per-neighbour combine is the identity; the shared engine supplies the tiling,
// the per-dim vectorised stencil, threading, and the D>4 gather fallback
// (`visit_nn_fallback_` / `reduce_fwd_fallback_`, re-exported for the tests).
//
//  visit_nn(l, body):    body(i, phi, nbrs_sum) -> void   (Σ all 2·ndims nbrs)
//  reduce_fwd(l, body):  body(phi, fwd_sum) -> T           (Σ ndims fwd nbrs)

namespace reticolo::action::detail {

template <class T, class Body>
inline void visit_nn(Lattice<T> const& l, Body&& body) noexcept {
    visit_stencil<T>(l, IdentityCombine{}, std::forward<Body>(body));
}

template <class T, class Acc = T, class Body>
[[nodiscard]] inline Acc reduce_fwd(Lattice<T> const& l, Body&& body) noexcept {
    return reduce_stencil<T, Acc>(l, IdentityCombine{}, std::forward<Body>(body));
}

}  // namespace reticolo::action::detail
