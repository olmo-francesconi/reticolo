#pragma once

#include <reticolo/action/sweep/stencil.hpp>
#include <reticolo/core/lattice.hpp>

#include <cstddef>
#include <utility>

// Site nearest-neighbour drivers — the identity-combine specialisation of the
// shared dimension-generic engine in <reticolo/action/sweep/stencil.hpp>. A
// site action consumes the raw neighbour sum (self + Σ neighbours), so the
// per-neighbour combine is the identity; the shared engine supplies the tiling,
// the per-dim (1..4) vectorised stencil, and threading. Neighbours are computed
// from the lattice strides — there is no neighbour table and no D>4 path.
//
//  visit_nn(l, body):    body(i, phi, nbrs_sum) -> void   (Σ all 2·ndims nbrs)
//  reduce_fwd(l, body):  body(phi, fwd_sum) -> T           (Σ ndims fwd nbrs)

namespace reticolo::action::sweep {

template <class T, class Body>
inline void visit_nn(Lattice<T> const& l, Body&& body) noexcept {
    visit_stencil<AllDirs, T>(l, IdentityCombine{}, std::forward<Body>(body));
}

template <class T, class Acc = T, class Body>
[[nodiscard]] inline Acc reduce_fwd(Lattice<T> const& l, Body&& body) noexcept {
    return reduce_stencil<FwdOnly, T, Acc>(l, IdentityCombine{}, std::forward<Body>(body));
}

}  // namespace reticolo::action::sweep
