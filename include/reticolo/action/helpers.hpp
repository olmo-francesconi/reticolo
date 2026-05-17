#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/site.hpp>

#include <cstddef>

// =============================================================================
//  Small helpers reused by the built-in scalar actions to avoid copy-pasting
//  the neighbour-loop boilerplate across `s_full`, `s_local`, `ds_local`.
//
//  Both helpers skip invalid neighbours (open BCs return
//  `Site::k_invalid_value` from `next`/`prev`), so the same body works on
//  periodic and open lattices without per-action `if constexpr`.
// =============================================================================

namespace reticolo::action {

// Sum of phi(y) over forward NN of x. Used in `s_full` (positive-mu bond
// convention).
template <class T>
[[nodiscard]] T fwd_neighbour_sum(Lattice<T> const& l, Site x) noexcept {
    T sum = T{0};
    for (std::size_t mu = 0; mu < l.ndims(); ++mu) {
        Site const fwd = l.next(x, mu);
        if (fwd.is_valid()) {
            sum += l[fwd];
        }
    }
    return sum;
}

// Sum of phi(y) over fwd + bwd NN of x. Used in `s_local` / `ds_local` where
// every bond touching x is counted once.
template <class T>
[[nodiscard]] T nn_neighbour_sum(Lattice<T> const& l, Site x) noexcept {
    T sum = T{0};
    for (std::size_t mu = 0; mu < l.ndims(); ++mu) {
        Site const fwd = l.next(x, mu);
        Site const bwd = l.prev(x, mu);
        if (fwd.is_valid()) {
            sum += l[fwd];
        }
        if (bwd.is_valid()) {
            sum += l[bwd];
        }
    }
    return sum;
}

}  // namespace reticolo::action
