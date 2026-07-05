#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/site.hpp>

#include <cstddef>

// Bond traversal drivers for the BondAction family (XY / planar rotor): the
// energy is a sum over bonds of a function of the two endpoint values, so unlike
// the site drivers the neighbour values can't be pre-summed — each bond is
// evaluated individually.

namespace reticolo::action::detail {

// Sum a bond kernel over the d FORWARD neighbours of every site (each bond once).
template <class T, class Acc, class Bond>
[[nodiscard]] inline Acc reduce_bonds_fwd(Lattice<T> const& l, Bond&& bond) noexcept {
    auto const& idx              = l.indexing_ref();
    T const* const data          = l.data();
    Site::value_type const* next = idx.next_data();
    std::size_t const n          = idx.nsites();
    std::size_t const d          = idx.ndims();

    Acc total{};
    for (std::size_t i = 0; i < n; ++i) {
        T const self           = data[i];
        std::size_t const base = i * d;
        for (std::size_t mu = 0; mu < d; ++mu) {
            total += static_cast<Acc>(bond(self, data[next[base + mu]]));
        }
    }
    return total;
}

// For each site, sum a bond kernel over ALL 2*ndims neighbours and hand the site
// index + accumulated sum to `sink` (which applies the scale and writes out).
template <class T, class Bond, class Sink>
inline void for_each_site_bond(Lattice<T> const& l, Bond&& bond, Sink&& sink) noexcept {
    auto const& idx              = l.indexing_ref();
    T const* const data          = l.data();
    Site::value_type const* next = idx.next_data();
    Site::value_type const* prev = idx.prev_data();
    std::size_t const n          = idx.nsites();
    std::size_t const d          = idx.ndims();

    for (std::size_t i = 0; i < n; ++i) {
        T const self           = data[i];
        T sum                  = T{0};
        std::size_t const base = i * d;
        for (std::size_t mu = 0; mu < d; ++mu) {
            sum += bond(self, data[next[base + mu]]);
            sum += bond(self, data[prev[base + mu]]);
        }
        sink(i, sum);
    }
}

}  // namespace reticolo::action::detail
