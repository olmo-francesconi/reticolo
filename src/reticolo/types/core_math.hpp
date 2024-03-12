/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/types/core_math.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <cstddef>

#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"
// #include "reticolo/types/core.hpp"

namespace reticolo {

/*--------------------------------------------------------------------------------------------------
    Math functions for core DataTypes
--------------------------------------------------------------------------------------------------*/

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

template <size_t dim>
inline auto dot(const intvect<dim>& vect) -> uint {
    uint Res = 0;
    for (uint Comp = 0; Comp < dim; Comp++) {
        Res += vect[Comp] * vect[Comp];
    }
    return Res;
}

template <size_t dim>
inline auto dot(const intvect<dim>& lhs, const intvect<dim>& rhs) -> uint {
    uint Res = 0;
    for (uint Comp = 0; Comp < dim; Comp++) {
        Res += lhs[Comp] * rhs[Comp];
    }
    return Res;
}

template <size_t dim>
inline auto operator+=(intvect<dim>& lhs, const intvect<dim>& rhs) -> intvect<dim>& {
    for (uint Comp = 0; Comp < dim; Comp++) {
        lhs[Comp] += rhs[Comp];
    }
    return lhs;
}

template <size_t dim>
inline auto operator-=(intvect<dim>& lhs, const intvect<dim>& rhs) -> intvect<dim>& {
    for (uint Comp = 0; Comp < dim; Comp++) {
        lhs[Comp] -= rhs[Comp];
    }
    return lhs;
}
template <size_t dim>
inline auto operator*=(intvect<dim>& lhs, const intvect<dim>& rhs) -> intvect<dim>& {
    for (uint Comp = 0; Comp < dim; Comp++) {
        lhs[Comp] *= rhs[Comp];
    }
    return lhs;
}

template <size_t dim>
inline auto operator/=(intvect<dim>& lhs, const intvect<dim>& rhs) -> intvect<dim>& {
    for (uint Comp = 0; Comp < dim; Comp++) {
        lhs[Comp] /= rhs[Comp];
    }
    return lhs;
}

template <size_t dim>
inline auto operator%=(intvect<dim>& lhs, const intvect<dim>& rhs) -> intvect<dim>& {
    for (uint Comp = 0; Comp < dim; Comp++) {
        lhs[Comp] %= rhs[Comp];
    }
    return lhs;
}

template <size_t dim>
inline auto operator+(intvect<dim> lhs, const intvect<dim>& rhs) -> intvect<dim> {
    lhs += rhs;
    return lhs;
}

template <size_t dim>
inline auto operator-(intvect<dim> lhs, const intvect<dim>& rhs) -> intvect<dim> {
    lhs -= rhs;
    return lhs;
}
template <size_t dim>
inline auto operator*(intvect<dim> lhs, const intvect<dim>& rhs) -> intvect<dim> {
    lhs *= rhs;
    return lhs;
}

template <size_t dim>
inline auto operator/(intvect<dim> lhs, const intvect<dim>& rhs) -> intvect<dim> {
    lhs /= rhs;
    return lhs;
}

template <size_t dim>
inline auto operator%(intvect<dim> lhs, const intvect<dim>& rhs) -> intvect<dim> {
    lhs %= rhs;
    return lhs;
}

template <size_t dim>
inline auto getVolume(const intvect<dim>& Sizes) -> int {
    int Res = 1;
    for (uint Comp = 0; Comp < dim; Comp++) {
        Res *= Sizes[Comp];
    }
    return Res;
}

template <size_t dim>
inline auto getCoord(uint site, const intvect<dim>& subvols) -> intvect<dim> {
    intvect<dim> Res;
    for (uint Comp = 0; Comp < dim; Comp++) {
        Res[Comp] = site / subvols[Comp];
        site = site % subvols[Comp];
    }
    return Res;
}

template <size_t dim>
inline void advance_coord(const intvect<dim>& sizes, intvect<dim>& coord) {
    coord.back()++;
    for (uint Dir = dim - 1; Dir > 0; Dir--) {
        if (coord[Dir] == sizes[Dir]) {
            coord[Dir] = 0;
            coord[Dir - 1]++;
        } else {
            return;
        }
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