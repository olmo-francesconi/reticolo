/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/types/core.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <H5Cpp.h>

#include <array>
#include <complex>
#include <cstddef>
#include <functional>
#include <numeric>

namespace reticolo {

// enumerator for logging modes
enum LOG_mode { silent = 0, log_only = 1, file_only = 2, all = 3 };

// enumerator for lattice dimensions
enum { _t = 0, _x = 1, _y = 2, _z = 3 };

// enumerator for lattice dimensions
enum {
    _x0 = 0,
    _x1 = 1,
    _x2 = 2,
    _x3 = 3,
    _x4 = 4,
    _x5 = 5,
    _x6 = 6,
    _x7 = 7,
    _x8 = 8,
    _x9 = 9,
};

// basic types
using uint = unsigned int;
using RealF = float;
using RealD = double;
using ComplexF = std::complex<float>;
using ComplexD = std::complex<double>;

// uint vector types
template <size_t dim>
using uintvect = std::array<uint, dim>;

template <size_t dim>
auto operator*(const uintvect<dim>& lhs, const uintvect<dim>& rhs) -> uint {
    uint Res = 0;
    for (int Comp = 0; Comp < dim; Comp++) {
        Res += lhs[Comp] * rhs[Comp];
    }
    return Res;
}

template <size_t dim>
inline void advance_coord(const uintvect<dim>& sizes, uintvect<dim>& coord) {
    coord.back()++;
    for (int Dir = dim - 1; Dir >= 0; Dir--) {
        if (coord[Dir] == sizes[Dir]) {
            coord[Dir] = 0;
            coord[Dir - 1]++;
        } else {
            return;
        }
    }
}

template <size_t dim>
inline auto get_volume(const uintvect<dim>& Sizes) -> uint {
    return std::accumulate(Sizes.begin(), Sizes.end(), 1, std::multiplies<>());
}

/*--------------------------------------------------------------------------------------------------
    Hdf5 Types for core DataTypes
--------------------------------------------------------------------------------------------------*/

template <typename T>
auto make_H5_Type() {}

template <>
auto make_H5_Type<uint>() {
    return H5::PredType::NATIVE_UINT;
}

template <>
auto make_H5_Type<RealF>() {
    return H5::PredType::NATIVE_FLOAT;
}

template <>
auto make_H5_Type<RealD>() {
    return H5::PredType::NATIVE_DOUBLE;
}

template <>
auto make_H5_Type<ComplexF>() {
    H5::CompType Type(sizeof(ComplexF));
    Type.insertMember("re", 0, H5::PredType::NATIVE_FLOAT);
    Type.insertMember("im", sizeof(ComplexF) / 2, H5::PredType::NATIVE_FLOAT);
    return Type;
}

template <>
auto make_H5_Type<ComplexD>() {
    H5::CompType Type(sizeof(ComplexD));
    Type.insertMember("re", 0, H5::PredType::NATIVE_DOUBLE);
    Type.insertMember("im", sizeof(ComplexD) / 2, H5::PredType::NATIVE_DOUBLE);
    return Type;
}

}  // namespace reticolo