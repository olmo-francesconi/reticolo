/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/types/core_math.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <cassert>
#include <numeric>
#include <vector>

#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"

namespace reticolo {

/*--------------------------------------------------------------------------------------------------
    Math functions for core DataTypes
--------------------------------------------------------------------------------------------------*/

/* Reset/Initialization methods */

template <RealValue T>
inline void reset(T& val) {
    val = 0.0;
}

template <ComplexValue T>
inline void reset(T& val) {
    val = {0.0, 0.0};
}

/* Dot and other Multiplication */

template <RealValue T>
inline auto dot(const T& val) -> T {
    return val * val;
}

template <RealValue T>
inline auto dot(const T& lhs, const T& rhs) -> T {
    return lhs * rhs;
}

template <ComplexValue T>
inline auto dot(const T& val) -> T::value_type {
    return val.real() * val.real() + val.imag() * val.imag();
}

template <ComplexValue T>
inline auto dot(const T& lhs, const T& rhs) -> T::value_type {
    return lhs.real() * rhs.real() + lhs.imag() * rhs.imag();
}

template <typename T>
inline auto dot(const std::vector<T>& vect) -> T {
    return std::accumulate(vect.begin(), vect.end(), 0, [](int sum, int value) { return sum + value * value; });
}

template <typename T>
inline auto dot(const std::vector<T>& vec1, const std::vector<T>& vec2) -> T {
    return std::inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0);
}

template <typename T>
inline void advance_coord(const std::vector<T>& sizes, std::vector<T>& coord) {
    assert(sizes.size() == coord.size());
    coord.back()++;
    for (uint Dir = sizes.size() - 1; Dir > 0; Dir--) {
        if (coord[Dir] == sizes[Dir]) {
            coord[Dir] = 0;
            coord[Dir - 1]++;
        } else {
            return;
        }
    }
    if (coord[0] == sizes[0]) {
        coord[0] = 0;
    }
}

/* Complex -> Real casting/projecting/quenching */
template <RealValue T>
inline auto make_real(const T& Var) -> T {
    return Var;
}

template <ComplexValue T>
inline auto make_real(const T& Var) -> T::value_type {
    return Var.real();
}
}  // namespace reticolo
