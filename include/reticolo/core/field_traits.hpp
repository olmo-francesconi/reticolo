#pragma once

#include <reticolo/core/cplx.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>

#include <complex>
#include <cstddef>

namespace reticolo {

// Real scalar underlying a field value type: `T` for real fields, the
// component type for `std::complex<T>`. MD kick coefficients, LLR window
// parameters, and other intrinsically-real quantities are typed through this
// so a complex field never forces a complex coefficient.
template <class T>
struct real_scalar {
    using type = T;
};
template <class T>
struct real_scalar<std::complex<T>> {
    using type = T;
};
// cplx<T> is the device-side complex used by DeviceField (the layout-compatible
// twin of std::complex<T>); it takes the same real underlying scalar.
template <class T>
struct real_scalar<cplx<T>> {
    using type = T;
};
template <class T>
using real_scalar_t = real_scalar<T>::type;

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

template <class G, class T>
[[nodiscard]] inline std::size_t flat_size(MatrixLinkLattice<G, T> const& f) noexcept {
    return f.ncomponents();
}

}  // namespace reticolo
