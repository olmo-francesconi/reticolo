/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/types/core.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <hdf5.h>

#include <array>
#include <complex>
#include <cstddef>

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

// Core types
using uint = unsigned int;
using RealF = float;
using RealD = double;
using ComplexF = std::complex<float>;
using ComplexD = std::complex<double>;

// uint vector types
template <size_t dim>
using intvect = std::array<int, dim>;

/*--------------------------------------------------------------------------------------------------
    Hdf5 Types for core DataTypes
--------------------------------------------------------------------------------------------------*/

template <typename T>
auto make_H5_Type();

template <>
auto make_H5_Type<uint>() {
    return H5T_NATIVE_UINT;
}

template <>
auto make_H5_Type<RealF>() {
    return H5T_NATIVE_FLOAT;
}

template <>
auto make_H5_Type<RealD>() {
    return H5T_NATIVE_DOUBLE;
}

template <>
auto make_H5_Type<ComplexF>() {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(ComplexF));
    H5Tinsert(DataTypeHid, "re", 0, H5T_NATIVE_FLOAT);
    H5Tinsert(DataTypeHid, "im", sizeof(ComplexD) / 2, H5T_NATIVE_FLOAT);
    return DataTypeHid;
}

template <>
auto make_H5_Type<ComplexD>() {
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(ComplexD));
    H5Tinsert(DataTypeHid, "re", 0, H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "im", sizeof(ComplexD) / 2, H5T_NATIVE_DOUBLE);
    return DataTypeHid;
}

}  // namespace reticolo