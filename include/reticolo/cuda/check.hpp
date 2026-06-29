#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

// CUDA error-checking discipline. Every CUDA / cuRAND / cub call goes through
// RETICOLO_CUDA_CHECK; every kernel launch is followed by
// RETICOLO_CUDA_CHECK_LAUNCH(). A CUDA call is a system boundary, so throwing
// on failure is the right amount of error handling — a silently-ignored device
// error would corrupt the Markov chain invisibly.

namespace reticolo::cuda {

inline void check_cuda(cudaError_t err, char const* expr, char const* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string{"CUDA error: "} + cudaGetErrorString(err) + " (" +
                                 expr + ") at " + file + ":" + std::to_string(line));
    }
}

}  // namespace reticolo::cuda

#define RETICOLO_CUDA_CHECK(expr) ::reticolo::cuda::check_cuda((expr), #expr, __FILE__, __LINE__)

// After a kernel launch: surface a launch-configuration / async error. In a
// debug build also synchronise so the error is attributed to this launch
// rather than surfacing at the next unrelated CUDA call.
#if defined(NDEBUG)
    #define RETICOLO_CUDA_CHECK_LAUNCH() RETICOLO_CUDA_CHECK(cudaGetLastError())
#else
    #define RETICOLO_CUDA_CHECK_LAUNCH()                 \
        do {                                             \
            RETICOLO_CUDA_CHECK(cudaGetLastError());     \
            RETICOLO_CUDA_CHECK(cudaDeviceSynchronize()); \
        } while (0)
#endif
