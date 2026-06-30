#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/pinned.hpp>
#include <reticolo/cuda/probes/selftest.hpp>

#include <cstddef>

namespace reticolo::cuda {

namespace {

__global__ void scale_kernel(double* d, std::size_t n, double a) {
    std::size_t const i = (static_cast<std::size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (i < n) {
        d[i] *= a;
    }
}

}  // namespace

bool selftest() {
    constexpr std::size_t n = 1024;
    constexpr double scale  = 2.0;

    PinnedBuffer<double> host{n};
    PinnedBuffer<double> back{n};
    for (std::size_t i = 0; i < n; ++i) {
        host.data()[i] = static_cast<double>(i);
    }

    DeviceBuffer<double> dev{n};
    dev.copy_from_host(host.data());

    constexpr unsigned block = 256;
    unsigned const grid      = static_cast<unsigned>((n + block - 1) / block);
    scale_kernel<<<grid, block>>>(dev.data(), n, scale);
    RETICOLO_CUDA_CHECK_LAUNCH();

    dev.copy_to_host(back.data());
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    for (std::size_t i = 0; i < n; ++i) {
        if (back.data()[i] != scale * static_cast<double>(i)) {
            return false;
        }
    }
    return true;
}

}  // namespace reticolo::cuda
