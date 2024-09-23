#pragma once

#include <H5Ipublic.h>
#include <H5Tpublic.h>

#include <complex>
#include <concepts>

#include "reticolo/core/tools/hdf5_helpers.hpp"  // IWYU pragma: keep
#include "reticolo/core/types/real.hpp"

namespace reticolo {
/*--------------------------------------------------------------------------------------------------
    Type definition
--------------------------------------------------------------------------------------------------*/
using ComplexF = std::complex<RealF>;
using ComplexD = std::complex<RealD>;

/*--------------------------------------------------------------------------------------------------
   concepts definition
--------------------------------------------------------------------------------------------------*/
template <typename T>
concept ComplexValue = std::same_as<T, ComplexD> || std::same_as<T, ComplexF>;

template <typename T>
concept isComplexF = std::same_as<T, ComplexF>;

template <typename T>
concept isComplexD = std::same_as<T, ComplexD>;

/*--------------------------------------------------------------------------------------------------
    reset()
--------------------------------------------------------------------------------------------------*/
template <ComplexValue T>
inline void reset(T& val) {
    val = {0.0, 0.0};
}

/*--------------------------------------------------------------------------------------------------
    addition
--------------------------------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------------------------------
    multiplications
--------------------------------------------------------------------------------------------------*/

/*--------------------------------------------------------------------------------------------------
    dot product
--------------------------------------------------------------------------------------------------*/
template <ComplexValue T>
inline auto dot(const T& val) -> T::value_type {
    return val.real() * val.real() + val.imag() * val.imag();
}

template <ComplexValue T>
inline auto dot(const T& lhs, const T& rhs) -> T::value_type {
    return lhs.real() * rhs.real() + lhs.imag() * rhs.imag();
}

/*--------------------------------------------------------------------------------------------------
    randomize
--------------------------------------------------------------------------------------------------*/
template <ComplexValue T, typename TDist, typename TGen>
inline void randomize(T& val, const typename T::value_type& scale, TDist& dist, TGen& rng)
    requires std::same_as<typename T::value_type, typename TDist::result_type>
{
    val = scale * T(dist(rng), dist(rng));
}

/*--------------------------------------------------------------------------------------------------
    Cast to reals
--------------------------------------------------------------------------------------------------*/
template <ComplexValue T>
inline auto make_real(const T& Var) -> T::value_type {
    return Var.real();
}

/*--------------------------------------------------------------------------------------------------
    Hdf5 Types
--------------------------------------------------------------------------------------------------*/
template <>
inline auto make_H5_Type<ComplexF>() -> hid_t {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(ComplexF));
    H5Tinsert(DataTypeHid, "re", 0, H5T_NATIVE_FLOAT);
    H5Tinsert(DataTypeHid, "im", sizeof(ComplexF) / 2, H5T_NATIVE_FLOAT);
    return DataTypeHid;
}

template <>
inline auto make_H5_Type<ComplexD>() -> hid_t {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(ComplexD));
    H5Tinsert(DataTypeHid, "re", 0, H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "im", sizeof(ComplexD) / 2, H5T_NATIVE_DOUBLE);
    return DataTypeHid;
}

}  // namespace reticolo