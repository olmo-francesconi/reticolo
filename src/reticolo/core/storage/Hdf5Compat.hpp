#pragma once

#include <H5public.h>

namespace reticolo::storage::hdf5 {

#ifdef RETICOLO_HDF5_VERSION_MAJOR
inline constexpr unsigned int version_major = RETICOLO_HDF5_VERSION_MAJOR;
#else
inline constexpr unsigned int version_major = H5_VERS_MAJOR;
#endif

#ifdef RETICOLO_HDF5_VERSION_MINOR
inline constexpr unsigned int version_minor = RETICOLO_HDF5_VERSION_MINOR;
#else
inline constexpr unsigned int version_minor = H5_VERS_MINOR;
#endif

#if defined(RETICOLO_HDF5_NATIVE_COMPLEX) && RETICOLO_HDF5_NATIVE_COMPLEX
inline constexpr bool native_complex_enabled = true;
#else
inline constexpr bool native_complex_enabled = false;
#endif

inline constexpr bool has_v2_api = version_major >= 2;

constexpr auto supports_native_complex() -> bool { return has_v2_api && native_complex_enabled; }

}  // namespace reticolo::storage::hdf5
