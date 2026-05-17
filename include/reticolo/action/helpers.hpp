#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/site.hpp>

#include <cstddef>

// =============================================================================
//  Small helpers reused by the built-in scalar actions to avoid copy-pasting
//  the neighbour-loop boilerplate across `s_full`, `s_local`, `ds_local`.
//
//  Each helper ships in two flavours:
//
//   * `_unchecked` — assumes every neighbour of `x` is valid. Safe to call
//     on any site of an all-periodic lattice and on any *bulk* site (returned
//     by `Lattice::bulk_sites()`) of an open-BC lattice. The inner loop is
//     branch-free, which keeps the hot kernel autovectorizable.
//
//   * (no suffix) — guards `next`/`prev` with `is_valid()`. Use for skin
//     sites under open BCs.
//
//  The recommended hot-loop pattern is therefore
//
//      for (Site x : l.bulk_sites()) { ...nn_neighbour_sum_unchecked(l, x)... }
//      for (Site x : l.skin_sites()) { ...nn_neighbour_sum(l, x)... }
//
//  Under all-periodic, `skin_sites()` is empty, so the safe branch is never
//  taken and the entire pass runs on the unchecked path.
// =============================================================================

namespace reticolo::action {

// Sum of phi(y) over forward NN of x; precondition: every fwd neighbour is
// valid (use on bulk sites / all-periodic lattices). Branch-free inner loop.
template <class T>
[[nodiscard]] T fwd_neighbour_sum_unchecked(Lattice<T> const& l, Site x) noexcept {
    T sum = T{0};
    for (std::size_t mu = 0; mu < l.ndims(); ++mu) {
        sum += l[l.next(x, mu)];
    }
    return sum;
}

// Sum of phi(y) over fwd + bwd NN of x; same precondition as above.
template <class T>
[[nodiscard]] T nn_neighbour_sum_unchecked(Lattice<T> const& l, Site x) noexcept {
    T sum = T{0};
    for (std::size_t mu = 0; mu < l.ndims(); ++mu) {
        sum += l[l.next(x, mu)];
        sum += l[l.prev(x, mu)];
    }
    return sum;
}

// Sum of phi(y) over forward NN of x, with open-BC guard. Use on skin sites.
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

// Sum of phi(y) over fwd + bwd NN of x, with open-BC guard.
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
