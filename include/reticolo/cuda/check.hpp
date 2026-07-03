#pragma once

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

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
// rather than surfacing at the next unrelated CUDA call — but NOT while a CUDA
// graph capture is in flight, where cudaDeviceSynchronize is illegal and would
// invalidate the capture. The HMC/LLR trajectory is captured with
// cudaStreamCaptureModeThreadLocal, so on the capturing thread the legacy
// stream reports Active; cudaStreamIsCapturing is a query and is itself legal
// during capture. This lets a `CMAKE_BUILD_TYPE=Debug` CUDA build run instead
// of aborting every capture.
#if defined(NDEBUG)
    #define RETICOLO_CUDA_CHECK_LAUNCH() RETICOLO_CUDA_CHECK(cudaGetLastError())
#else
    #define RETICOLO_CUDA_CHECK_LAUNCH()                                                           \
        do {                                                                                       \
            RETICOLO_CUDA_CHECK(cudaGetLastError());                                               \
            cudaStreamCaptureStatus reticolo_cap_status__ = cudaStreamCaptureStatusNone;           \
            RETICOLO_CUDA_CHECK(cudaStreamIsCapturing(cudaStreamLegacy, &reticolo_cap_status__));  \
            if (reticolo_cap_status__ == cudaStreamCaptureStatusNone) {                            \
                RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());                                      \
            }                                                                                      \
        } while (0)
#endif
