#pragma once

// Device momentum sampler — nvcc-only (.cuh; launches a kernel).
//
// Fills out[0..n) with standard normals = philox_normal2(seed, *traj, pair)
// using the SHARED HD Philox primitive (core/philox.hpp) — the same generator
// the CPU PhiloxRng runs. Embarrassingly parallel (one thread per output pair,
// no shared state). The trajectory counter is read from a DEVICE pointer, never
// a baked kernel literal, so a captured graph replays with a fresh counter just
// by advancing the device-side value.

#include <reticolo/core/philox.hpp>
#include <reticolo/cuda/check.hpp>

#include <cstdint>

#include <cuda_runtime.h>

namespace reticolo::cuda {

// Templated so the header-defined kernel gets weak/mergeable linkage (the
// same reason stencil_kernel<F> is a template): a plain __global__ in a header
// included by multiple TUs collides at device link under -rdc=true.
template <class T = double>
__global__ void
fill_normals_kernel(T* out, long n, std::uint64_t seed, std::uint64_t const* traj, double sigma) {
    long const pair    = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    long const n_pairs = (n + 1) / 2;
    if (pair >= n_pairs) {
        return;
    }
    double n0 = 0.0;
    double n1 = 0.0;
    philox_normal2(seed, *traj, static_cast<std::uint64_t>(pair), n0, n1);
    out[2 * pair] = static_cast<T>(n0 * sigma);
    if (((2 * pair) + 1) < n) {
        out[(2 * pair) + 1] = static_cast<T>(n1 * sigma);
    }
}

// Fill out[0..n) with N(0, sigma²). sigma defaults to 1 (momentum sampling); the
// LLR gauge hot-start passes sigma > 1 to disorder a link field before warm-in.
template <class T>
inline void fill_normals(T* out,
                         long n,
                         std::uint64_t seed,
                         std::uint64_t const* traj_dev,
                         cudaStream_t stream = nullptr,
                         double sigma        = 1.0) {
    if (n <= 0) {
        return;
    }
    long const n_pairs   = (n + 1) / 2;
    constexpr int kBlock = 256;
    auto const grid      = static_cast<unsigned>((n_pairs + kBlock - 1) / kBlock);
    fill_normals_kernel<T><<<grid, kBlock, 0, stream>>>(out, n, seed, traj_dev, sigma);
    RETICOLO_CUDA_CHECK_LAUNCH();
}

}  // namespace reticolo::cuda
