#pragma once

// Fills the banner's `gpu` row with live device details. Core cannot query the
// device (the one-way core ← cuda rule keeps <cuda_runtime.h> out of core), so
// this header — nvcc-only, host-callable — sets log::impl::gpu_banner_hook()
// at load time via a [[gnu::constructor]]. Any app that links the CUDA backend
// pulls this in through <reticolo/cuda/cuda.hpp>, so the gpu row appears in the
// banner with no per-app wiring. The actual cudaGetDeviceProperties call runs
// lazily when banner() invokes the hook (at log::start), by which point the
// CUDA context is up; on any error it returns "" and banner() omits the row.

#include <reticolo/core/log/log.hpp>

#include <format>
#include <string>

#include <cuda_runtime.h>

namespace reticolo::cuda::impl {

inline std::string device_banner_line() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) {
        return {};
    }
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) {
        return {};
    }
    int runtime = 0;
    cudaRuntimeGetVersion(&runtime);

    constexpr double bytes_per_gib = 1024.0 * 1024.0 * 1024.0;
    double const gib               = static_cast<double>(prop.totalGlobalMem) / bytes_per_gib;

    return std::format("{} · sm_{}{} · {:.0f} GiB · {} SM (CUDA {}.{})",
                       prop.name,
                       prop.major,
                       prop.minor,
                       gib,
                       prop.multiProcessorCount,
                       runtime / 1000,
                       (runtime % 1000) / 10);
}

// nvcc toolkit version this TU was compiled with — a compile-time constant from
// the CUDA predefined macros, which only exist in an nvcc .cu TU (hence it lives
// here, not in core/build_info.hpp). Feeds the `compiler` row: nvcc X.Y (host).
inline std::string nvcc_version() {
#if defined(__CUDACC_VER_MAJOR__)
    return std::format("{}.{}", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__);
#else
    return {};
#endif
}

// Runs at load time, before main() and thus before log::start()/banner(). The
// [[gnu::used]] forces the inline function to be emitted so the constructor is
// actually scheduled even though nothing references it. Idempotent across TUs.
[[gnu::used, gnu::constructor]] inline void register_device_banner() {
    log::impl::gpu_banner_hook()  = &device_banner_line;
    log::impl::nvcc_banner_hook() = &nvcc_version;
}

}  // namespace reticolo::cuda::impl
