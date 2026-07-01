#pragma once

// Deterministic device reductions / vector ops, header-only (.cuh — nvcc-only,
// kernels live here as in reduce_fwd.cuh). Fixed launch config so the summation
// order is reproducible run-to-run — required for HMC reversibility (a varying
// reduction tree changes ΔH non-deterministically). A simple block-partial +
// fixed-order host finish; device-scalar (_into) variants build on the same
// primitive and avoid a host sync inside the graph.
//
// Pointers are device pointers. The host launchers are `inline`; the kernels and
// host helpers sit in an anonymous namespace (one copy per including TU — no ODR
// clash, the same shape reduce_fwd.cuh already relies on).

#include <reticolo/cuda/check.hpp>

#include <vector>

#include <cuda_runtime.h>

namespace reticolo::cuda {

namespace {

constexpr int kBlock   = 256;
constexpr int kMaxGrid = 1024;

[[nodiscard]] inline int grid_for(long n) {
    long g = (n + kBlock - 1) / kBlock;
    if (g > kMaxGrid) {
        g = kMaxGrid;
    }
    return static_cast<int>(g);
}

// axpy in the field type T (f64 or f32): the MD drift/kick runs in field
// precision, matching the CPU mixed-precision HMC.
template <class T>
__global__ void axpy_kernel(T a, T const* x, T* y, long n) {
    long const stride = static_cast<long>(gridDim.x) * blockDim.x;
    for (long i = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x; i < n; i += stride) {
        y[i] += a * x[i];
    }
}

// One partial sum per block: grid-stride accumulate, then a fixed in-block tree
// reduction. With a fixed grid the set of partials is reproducible, and the
// host finish sums them in index order — so the whole reduction is deterministic.
// Input type `In` may be f32 (the field) but the accumulator is ALWAYS double:
// a float volume sum loses ~log2(V) bits and corrupts the ΔH the HMC accept
// depends on, exactly as on the CPU.
template <class In>
__global__ void block_sum_kernel(In const* x, long n, double* partial) {
    __shared__ double s[kBlock];
    long const stride = static_cast<long>(gridDim.x) * blockDim.x;
    double acc        = 0.0;
    for (long i = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x; i < n; i += stride) {
        acc += static_cast<double>(x[i]);
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            s[threadIdx.x] += s[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = s[0];
    }
}

// As block_sum_kernel but accumulating x[i]² in double — the kinetic-energy
// reduction. Each square is promoted to double before accumulation.
template <class In>
__global__ void block_sumsq_kernel(In const* x, long n, double* partial) {
    __shared__ double s[kBlock];
    long const stride = static_cast<long>(gridDim.x) * blockDim.x;
    double acc        = 0.0;
    for (long i = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x; i < n; i += stride) {
        double const v = static_cast<double>(x[i]);
        acc += v * v;
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            s[threadIdx.x] += s[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial[blockIdx.x] = s[0];
    }
}

// Single-block final pass: sum `count` block-partials into out[0], on device.
// Deterministic given a fixed grid / blockDim (the grid-stride load order and
// the tree are fixed). Run-to-run reproducible — reversibility requires it.
__global__ void final_reduce_kernel(double const* partials, int count, double* out) {
    __shared__ double s[kBlock];
    double acc = 0.0;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        acc += partials[i];
    }
    s[threadIdx.x] = acc;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            s[threadIdx.x] += s[threadIdx.x + off];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[0] = s[0];
    }
}

// Shared host finish: copy block partials back and sum in index order
// (reproducible), then free the device scratch on `stream`.
[[nodiscard]] inline double finish_partials(double* d_partial, int grid, cudaStream_t stream) {
    std::vector<double> partials(static_cast<std::size_t>(grid));
    RETICOLO_CUDA_CHECK(cudaMemcpyAsync(partials.data(),
                                        d_partial,
                                        static_cast<std::size_t>(grid) * sizeof(double),
                                        cudaMemcpyDeviceToHost,
                                        stream));
    RETICOLO_CUDA_CHECK(cudaFreeAsync(d_partial, stream));
    RETICOLO_CUDA_CHECK(cudaStreamSynchronize(stream));

    double total = 0.0;
    for (double v : partials) {
        total += v;  // fixed index order → reproducible
    }
    return total;
}

[[nodiscard]] inline double* alloc_partials(int grid, cudaStream_t stream) {
    double* d_partial = nullptr;
    RETICOLO_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&d_partial),
                                        static_cast<std::size_t>(grid) * sizeof(double),
                                        stream));
    return d_partial;
}

}  // namespace

// y[i] += a * x[i], in the field type. The f32 overload supports mixed-precision
// MD (drift/kick run in field precision); the volume reductions still accumulate
// in double (see below).
inline void axpy_f64(double a, double const* x, double* y, long n, cudaStream_t stream = nullptr) {
    if (n <= 0) {
        return;
    }
    axpy_kernel<double><<<grid_for(n), kBlock, 0, stream>>>(a, x, y, n);
    RETICOLO_CUDA_CHECK_LAUNCH();
}

inline void axpy_f32(float a, float const* x, float* y, long n, cudaStream_t stream = nullptr) {
    if (n <= 0) {
        return;
    }
    axpy_kernel<float><<<grid_for(n), kBlock, 0, stream>>>(a, x, y, n);
    RETICOLO_CUDA_CHECK_LAUNCH();
}

// Σ x[i], accumulated in double with a fixed, reproducible order.
[[nodiscard]] inline double reduce_sum_f64(double const* x, long n, cudaStream_t stream = nullptr) {
    if (n <= 0) {
        return 0.0;
    }
    int const grid    = grid_for(n);
    double* d_partial = alloc_partials(grid, stream);
    block_sum_kernel<double><<<grid, kBlock, 0, stream>>>(x, n, d_partial);
    RETICOLO_CUDA_CHECK_LAUNCH();
    return finish_partials(d_partial, grid, stream);
}

// Σ x[i]², same fixed-order determinism. The HMC kinetic term ½Σp² — kept a
// separate primitive (rather than square-into-scratch + reduce_sum) so the
// reduction stays one pass and run-to-run reproducible for reversibility. The
// f32 overload reads float momenta but accumulates the sum of squares in double.
[[nodiscard]] inline double
reduce_sumsq_f64(double const* x, long n, cudaStream_t stream = nullptr) {
    if (n <= 0) {
        return 0.0;
    }
    int const grid    = grid_for(n);
    double* d_partial = alloc_partials(grid, stream);
    block_sumsq_kernel<double><<<grid, kBlock, 0, stream>>>(x, n, d_partial);
    RETICOLO_CUDA_CHECK_LAUNCH();
    return finish_partials(d_partial, grid, stream);
}

[[nodiscard]] inline double
reduce_sumsq_f32(float const* x, long n, cudaStream_t stream = nullptr) {
    if (n <= 0) {
        return 0.0;
    }
    int const grid    = grid_for(n);
    double* d_partial = alloc_partials(grid, stream);
    block_sumsq_kernel<float><<<grid, kBlock, 0, stream>>>(x, n, d_partial);
    RETICOLO_CUDA_CHECK_LAUNCH();
    return finish_partials(d_partial, grid, stream);
}

// Capacity a `partials` scratch buffer must have for the *_into reductions.
inline constexpr long k_reduce_max_grid = 1024;

// Device-scalar reductions for the HMC hot loop: write the result to out[0] on
// the device (no host sync, no per-call allocation). `partials` is a caller-
// owned scratch buffer of at least k_reduce_max_grid doubles. A fixed-config
// block reduction + a single-block final pass — deterministic run-to-run, which
// reversibility requires. The whole trajectory enqueues these on one stream and
// syncs once (vs the four malloc+sync round-trips reduce_sum/sumsq_f64 cost).
inline void reduce_sum_into(
    double* out, double const* x, long n, double* partials, cudaStream_t stream = nullptr) {
    if (n <= 0) {
        return;
    }
    int const grid = grid_for(n);
    block_sum_kernel<double><<<grid, kBlock, 0, stream>>>(x, n, partials);
    RETICOLO_CUDA_CHECK_LAUNCH();
    final_reduce_kernel<<<1, kBlock, 0, stream>>>(partials, grid, out);
    RETICOLO_CUDA_CHECK_LAUNCH();
}

inline void reduce_sumsq_into(
    double* out, double const* x, long n, double* partials, cudaStream_t stream = nullptr) {
    if (n <= 0) {
        return;
    }
    int const grid = grid_for(n);
    block_sumsq_kernel<double><<<grid, kBlock, 0, stream>>>(x, n, partials);
    RETICOLO_CUDA_CHECK_LAUNCH();
    final_reduce_kernel<<<1, kBlock, 0, stream>>>(partials, grid, out);
    RETICOLO_CUDA_CHECK_LAUNCH();
}

// f32 momenta overload: reads float, accumulates the sum of squares in double.
inline void reduce_sumsq_into(
    double* out, float const* x, long n, double* partials, cudaStream_t stream = nullptr) {
    if (n <= 0) {
        return;
    }
    int const grid = grid_for(n);
    block_sumsq_kernel<float><<<grid, kBlock, 0, stream>>>(x, n, partials);
    RETICOLO_CUDA_CHECK_LAUNCH();
    final_reduce_kernel<<<1, kBlock, 0, stream>>>(partials, grid, out);
    RETICOLO_CUDA_CHECK_LAUNCH();
}

}  // namespace reticolo::cuda
