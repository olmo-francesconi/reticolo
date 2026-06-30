#include <reticolo/core/philox.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/probes/rng_probe.hpp>
#include <reticolo/cuda/rng_philox.cuh>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

// Phase 2c gates for the device Philox momentum sampler.

namespace reticolo::cuda {

namespace {

__global__ void uniforms_kernel(double* out, int k, std::uint64_t seed, std::uint64_t traj) {
    int const i = (static_cast<int>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (i >= k) {
        return;
    }
    double u0 = 0.0;
    double u1 = 0.0;
    philox_uniform2(seed, traj, static_cast<std::uint64_t>(i), u0, u1);
    out[2 * i]       = u0;
    out[(2 * i) + 1] = u1;
}

}  // namespace

bool philox_host_matches_device() {
    constexpr int k              = 256;
    constexpr std::uint64_t seed = 0xABCDEFULL;
    constexpr std::uint64_t traj = 3;

    std::vector<double> host(static_cast<std::size_t>(2 * k));
    for (int i = 0; i < k; ++i) {
        philox_uniform2(seed,
                        traj,
                        static_cast<std::uint64_t>(i),
                        host[static_cast<std::size_t>(2 * i)],
                        host[static_cast<std::size_t>((2 * i) + 1)]);
    }

    DeviceBuffer<double> dev{static_cast<std::size_t>(2 * k)};
    constexpr int kBlock = 256;
    uniforms_kernel<<<(k + kBlock - 1) / kBlock, kBlock>>>(dev.data(), k, seed, traj);
    RETICOLO_CUDA_CHECK_LAUNCH();

    std::vector<double> back(static_cast<std::size_t>(2 * k));
    dev.copy_to_host(back.data());
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    for (std::size_t i = 0; i < host.size(); ++i) {
        if (host[i] != back[i]) {  // bit-identical, not approximate
            return false;
        }
    }
    return true;
}

bool philox_traj_distinct() {
    constexpr long n             = 1024;
    constexpr std::uint64_t seed = 0x1234ULL;

    DeviceBuffer<std::uint64_t> traj{1};
    DeviceBuffer<double> a{static_cast<std::size_t>(n)};
    DeviceBuffer<double> b{static_cast<std::size_t>(n)};
    DeviceBuffer<double> a2{static_cast<std::size_t>(n)};

    std::uint64_t t0 = 0;
    std::uint64_t t1 = 1;
    traj.copy_from_host(&t0);
    fill_normals(a.data(), n, seed, traj.data());
    traj.copy_from_host(&t1);
    fill_normals(b.data(), n, seed, traj.data());
    traj.copy_from_host(&t0);
    fill_normals(a2.data(), n, seed, traj.data());

    std::vector<double> ha(static_cast<std::size_t>(n));
    std::vector<double> hb(static_cast<std::size_t>(n));
    std::vector<double> ha2(static_cast<std::size_t>(n));
    a.copy_to_host(ha.data());
    b.copy_to_host(hb.data());
    a2.copy_to_host(ha2.data());
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    bool same_t0 = true;
    bool diff_t1 = false;
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
        if (ha[i] != ha2[i]) {
            same_t0 = false;
        }
        if (ha[i] != hb[i]) {
            diff_t1 = true;
        }
    }
    return same_t0 && diff_t1;
}

bool philox_moments_ok() {
    constexpr long n             = 1000000;
    constexpr std::uint64_t seed = 0x77ULL;

    DeviceBuffer<std::uint64_t> traj{1};
    std::uint64_t t = 0;
    traj.copy_from_host(&t);

    DeviceBuffer<double> d{static_cast<std::size_t>(n)};
    fill_normals(d.data(), n, seed, traj.data());

    std::vector<double> h(static_cast<std::size_t>(n));
    d.copy_to_host(h.data());
    RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());

    double sum   = 0.0;
    double sumsq = 0.0;
    for (double const x : h) {
        sum += x;
        sumsq += x * x;
    }
    double const mean = sum / static_cast<double>(n);
    double const var  = (sumsq / static_cast<double>(n)) - (mean * mean);
    return (std::abs(mean) < 0.01) && (std::abs(var - 1.0) < 0.01);
}

}  // namespace reticolo::cuda
