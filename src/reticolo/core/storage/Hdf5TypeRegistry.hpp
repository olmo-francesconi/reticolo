#pragma once

#include <H5Ipublic.h>
#include <H5Tpublic.h>

#include <cstddef>

namespace reticolo {

template <typename T>
inline auto make_H5_Type() -> hid_t;

template <>
inline auto make_H5_Type<unsigned short>() -> hid_t {
    return H5T_NATIVE_USHORT;
}

template <>
inline auto make_H5_Type<unsigned int>() -> hid_t {
    return H5T_NATIVE_UINT;
}

template <>
inline auto make_H5_Type<size_t>() -> hid_t {
    return H5T_NATIVE_ULONG;
}

}  // namespace reticolo
