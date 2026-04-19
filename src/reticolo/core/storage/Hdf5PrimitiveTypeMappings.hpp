#pragma once

#include <H5Ipublic.h>
#include <H5Tpublic.h>

#include <array>

#include "reticolo/core/storage/Hdf5Compat.hpp"
#include "reticolo/core/storage/Hdf5TypeRegistry.hpp"
#include "reticolo/core/types/complex.hpp"
#include "reticolo/core/types/hfield.hpp"
#include "reticolo/core/types/real.hpp"

namespace reticolo {

template <>
inline auto make_H5_Type<RealF>() -> hid_t {
    return H5T_NATIVE_FLOAT;
}

template <>
inline auto make_H5_Type<RealD>() -> hid_t {
    return H5T_NATIVE_DOUBLE;
}

template <>
inline auto make_H5_Type<ComplexF>() -> hid_t {
#if defined(RETICOLO_HDF5_NATIVE_COMPLEX) && RETICOLO_HDF5_NATIVE_COMPLEX
    if constexpr (storage::hdf5::supports_native_complex()) {
        return H5Tcomplex_create(H5T_NATIVE_FLOAT);
    }
#endif
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(ComplexF));
    H5Tinsert(DataTypeHid, "re", 0, H5T_NATIVE_FLOAT);
    H5Tinsert(DataTypeHid, "im", sizeof(ComplexF) / 2, H5T_NATIVE_FLOAT);
    return DataTypeHid;
}

template <>
inline auto make_H5_Type<ComplexD>() -> hid_t {
#if defined(RETICOLO_HDF5_NATIVE_COMPLEX) && RETICOLO_HDF5_NATIVE_COMPLEX
    if constexpr (storage::hdf5::supports_native_complex()) {
        return H5Tcomplex_create(H5T_NATIVE_DOUBLE);
    }
#endif
    hid_t DataTypeHid = H5Tcreate(H5T_COMPOUND, sizeof(ComplexD));
    H5Tinsert(DataTypeHid, "re", 0, H5T_NATIVE_DOUBLE);
    H5Tinsert(DataTypeHid, "im", sizeof(ComplexD) / 2, H5T_NATIVE_DOUBLE);
    return DataTypeHid;
}

template <>
inline auto make_H5_Type<HField<RealF>>() -> hid_t {
    std::array<hsize_t, 1> Dims = {10};
    return H5Tarray_create(H5T_NATIVE_FLOAT, 1, Dims.data());
}

template <>
inline auto make_H5_Type<HField<RealD>>() -> hid_t {
    std::array<hsize_t, 1> Dims = {10};
    return H5Tarray_create(H5T_NATIVE_DOUBLE, 1, Dims.data());
}

}  // namespace reticolo
