#pragma once

#include <reticolo/core/exec/nn_stencil.hpp>
#include <reticolo/core/field/lattice.hpp>

#include <cstddef>
#include <utility>

// Site nearest-neighbour drivers — the identity-combine specialisation of the
// shared dimension-generic engine in <reticolo/core/exec/nn_stencil.hpp>. A
// site action consumes the raw neighbour sum (self + Σ neighbours), so the
// per-neighbour combine is the identity; the shared engine supplies the tiling,
// the per-dim (1..4) vectorised stencil, and threading. Neighbours are computed
// from the lattice strides — there is no neighbour table and no D>4 path.
//
//  nn_visit_all(l, body):    body(i, phi, nbrs_sum) -> void   (Σ all 2·ndims nbrs)
//  nn_reduce_fwd(l, body):  body(phi, fwd_sum) -> T           (Σ ndims fwd nbrs)

namespace reticolo::exec {

template <class T, class Body>
inline void nn_visit_all(Lattice<T> const& l, Body&& body) noexcept {
    nn_visit<AllDirs, T>(l, IdentityCombine{}, std::forward<Body>(body));
}

template <class T, class Acc = T, class Body>
[[nodiscard]] inline Acc nn_reduce_fwd(Lattice<T> const& l, Body&& body) noexcept {
    return nn_reduce<FwdOnly, T, Acc>(l, IdentityCombine{}, std::forward<Body>(body));
}

}  // namespace reticolo::exec
