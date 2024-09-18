#pragma once

#include <complex>
#include <concepts>

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

}  // namespace reticolo