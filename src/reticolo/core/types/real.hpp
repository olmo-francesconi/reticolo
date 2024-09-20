#pragma once
#include <H5Ipublic.h>
#include <H5Tpublic.h>

#include <concepts>

#include "reticolo/core/tools/hdf5_helpers.hpp"  // IWYU pragma: keep

namespace reticolo {
/*--------------------------------------------------------------------------------------------------
    Type definition
--------------------------------------------------------------------------------------------------*/
using RealF = float;
using RealD = double;

/*--------------------------------------------------------------------------------------------------
   concepts definition
--------------------------------------------------------------------------------------------------*/
template <typename T>
concept RealValue = std::same_as<T, RealD> || std::same_as<T, RealF>;

template <typename T>
concept isRealF = std::same_as<T, RealF>;

template <typename T>
concept isRealD = std::same_as<T, RealD>;

/*--------------------------------------------------------------------------------------------------
    reset()
--------------------------------------------------------------------------------------------------*/
template <RealValue T>
inline void reset(T& val) {
    val = 0.0;
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
template <RealValue T>
inline auto dot(const T& val) -> T {
    return val * val;
}

template <RealValue T>
inline auto dot(const T& lhs, const T& rhs) -> T {
    return lhs * rhs;
}

/*--------------------------------------------------------------------------------------------------
    randomize
--------------------------------------------------------------------------------------------------*/
template <RealValue T, typename TDist, typename TGen>
inline void randomize(T& val, const T& scale, TDist& dist, TGen& rng)
    requires std::same_as<T, typename TDist::result_type>
{
    val = scale * dist(rng);
}

/*--------------------------------------------------------------------------------------------------
    Cast to reals
--------------------------------------------------------------------------------------------------*/
template <RealValue T>
inline auto make_real(const T& Var) -> T {
    return Var;
}

/*--------------------------------------------------------------------------------------------------
    Hdf5 Types
--------------------------------------------------------------------------------------------------*/
template <>
inline auto make_H5_Type<RealF>() -> hid_t {
    return H5T_NATIVE_FLOAT;
}

template <>
inline auto make_H5_Type<RealD>() -> hid_t {
    return H5T_NATIVE_DOUBLE;
}

}  // namespace reticolo