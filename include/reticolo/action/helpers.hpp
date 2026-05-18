#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/site.hpp>

#include <cstddef>

// =============================================================================
//  Small helpers reused by the built-in scalar actions for the per-site
//  Metropolis kernels (`s_local`, `ds_local`). Hot kernels (`s_full`,
//  `compute_force`) hoist `data` + `next` + `prev` raw pointers once and
//  inline the neighbour loop directly — see `action/builtins/phi4.hpp` for
//  the canonical pattern. Don't use these helpers from inside those hot
//  paths; the function-call boundary inhibits the compiler's hoist.
//
//  The library is periodic-only at present, so neighbour accesses never need
//  a validity check.
// =============================================================================

namespace reticolo::action {

// Sum of phi(y) over fwd + bwd NN of x. Used in `s_local` / `ds_local` where
// every bond touching x is counted once.
template <class T>
[[nodiscard]] T nn_neighbour_sum(Lattice<T> const& l, Site x) noexcept {
    T sum = T{0};
    for (std::size_t mu = 0; mu < l.ndims(); ++mu) {
        sum += l[l.next(x, mu)];
        sum += l[l.prev(x, mu)];
    }
    return sum;
}

}  // namespace reticolo::action
