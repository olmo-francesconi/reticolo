#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/link_lattice.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>

#include <cstddef>

namespace reticolo {

// Generic flat-element count for the field storage types. Lets algorithm
// code (HMC, integrators) loop over the underlying buffer without branching
// on which field it has: Lattice<T> reports nsites(); LinkLattice<T> reports
// ndim * nsites(); MatrixLinkLattice<G,T> reports ndim * 2N² * nsites (the
// raw real-component count). One overload per field — picked at compile
// time by ADL.

template <class T>
[[nodiscard]] inline std::size_t flat_size(Lattice<T> const& f) noexcept {
    return f.nsites();
}

template <class T>
[[nodiscard]] inline std::size_t flat_size(LinkLattice<T> const& f) noexcept {
    return f.nlinks();
}

template <class G, class T>
[[nodiscard]] inline std::size_t flat_size(MatrixLinkLattice<G, T> const& f) noexcept {
    return f.ncomponents();
}

}  // namespace reticolo
