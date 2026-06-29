#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/reduce.hpp>

#include <cuda_runtime.h>

#include <vector>

namespace reticolo::cuda {

namespace {

constexpr int kBlock   = 256;
constexpr int kMaxGrid = 1024;

[[nodiscard]] int grid_for(long n) {
    long g = (n + kBlock - 1) / kBlock;
    if (g > kMaxGrid) {
        g = kMaxGrid;
    }
    return static_cast<int>(g);
}

__global__ void axpy_kernel(double a, double const* x, double* y, long n) {
    long const stride = static_cast<long>(gridDim.x) * blockDim.x;
    for (long i = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x; i < n; i += stride) {
        y[i] += a * x[i];
    }
}

// One partial sum per block: grid-stride accumulate, then a fixed in-block tree
// reduction. With a fixed grid the set of partials is reproducible, and the
// host finish sums them in index order — so the whole reduction is deterministic.
__global__ void block_sum_kernel(double const* x, long n, double* partial) {
    __shared__ double s[kBlock];
    long const stride = static_cast<long>(gridDim.x) * blockDim.x;
    double acc        = 0.0;
    for (long i = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x; i < n; i += stride) {
        acc += x[i];
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

}  // namespace

void axpy_f64(double a, double const* x, double* y, long n, cudaStream_t stream) {
    if (n <= 0) {
        return;
    }
    axpy_kernel<<<grid_for(n), kBlock, 0, stream>>>(a, x, y, n);
    RETICOLO_CUDA_CHECK_LAUNCH();
}

double reduce_sum_f64(double const* x, long n, cudaStream_t stream) {
    if (n <= 0) {
        return 0.0;
    }
    int const grid = grid_for(n);

    double* d_partial = nullptr;
    RETICOLO_CUDA_CHECK(
        cudaMallocAsync(reinterpret_cast<void**>(&d_partial), static_cast<std::size_t>(grid) * sizeof(double), stream));

    block_sum_kernel<<<grid, kBlock, 0, stream>>>(x, n, d_partial);
    RETICOLO_CUDA_CHECK_LAUNCH();

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

}  // namespace reticolo::cuda
