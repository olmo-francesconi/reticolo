#pragma once

// Device-side fused observable reduction — the GPU twin of `obs::reduce`
// (obs/reduce.hpp). Runs a PACK of per-site kernels in ONE pass over a
// `DeviceField`, accumulating an independent double lane per kernel, so N
// observables cost one read of VRAM instead of N. The device previously had only
// two hardcoded reductions (`reduce_sum_f64`, `reduce_sumsq_f64`): a mean plus a
// second moment meant two full passes and nothing else was measurable at all.
//
// Dispatch is by ADL on the field type — the seam the integrator atoms already
// use (`drift_field`/`kick_add`). `obs::reduce` on a host `Lattice` and
// `cuda::reduce` on a `DeviceField` are SEPARATE overloads in separate
// namespaces; there is deliberately no dispatcher inside `obs::` that names
// `reduce` with a dependent argument, because such a body would bind names
// differently in a .cu TU than in a .cpp TU (IFNDR — and both TU kinds link into
// every CUDA test binary). Call it unqualified, or as `cuda::reduce`.
//
// Kernels are the SAME objects the CPU path uses: `obs::kernel::*` are
// RETICOLO_HD, so `phi_sq` here and `phi_sq` there are one definition.
//
// Determinism: fixed launch config (block and grid cap below), a grid-stride
// loop, per-lane block partials, and a host finish that folds the partials in
// index order — reproducible run-to-run for a fixed field size, exactly like
// reduce.cuh. It is NOT bit-identical to the CPU fold (different tree), which is
// the project's standing CPU↔GPU position: same ensemble, not same chain.
//
// Scope: homogeneous double lanes over a scalar field. Complex/heterogeneous
// lanes would need a wider accumulator and are deliberately not attempted here.
//
// This header declares its OWN launch constants rather than reusing reduce.cuh's
// `kBlock`/`kMaxGrid`/`grid_for`: those live in an anonymous namespace, so
// naming them from an `inline` function at namespace scope is what makes
// reduce.cuh's launchers technically IFNDR. Not propagating that.

#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/device_field.hpp>

#include <cstddef>
#include <tuple>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace reticolo::cuda {

inline constexpr int k_obs_block    = 256;
inline constexpr int k_obs_max_grid = 1024;

[[nodiscard]] inline int obs_grid_for(long n) {
    long g = (n + k_obs_block - 1) / k_obs_block;
    if (g > k_obs_max_grid) {
        g = k_obs_max_grid;
    }
    if (g < 1) {
        g = 1;
    }
    return static_cast<int>(g);
}

// One block partial per (lane, block): partials[(lane * gridDim.x) + blockIdx.x].
// The kernel pack arrives by value — every `obs::kernel::*` is an empty,
// trivially-copyable functor, so this costs nothing in the parameter bank.
template <int NLanes, class T, class... Ks>
__global__ void
obs_reduce_kernel(T const* __restrict__ x, long n, double* __restrict__ partials, Ks... ks) {
    __shared__ double sm[k_obs_block];  // NOLINT(cppcoreguidelines-avoid-c-arrays)

    double acc[NLanes] = {};  // NOLINT(cppcoreguidelines-avoid-c-arrays)
    for (long i = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x; i < n;
         i += static_cast<long>(gridDim.x) * blockDim.x) {
        T const self = x[i];
        int lane     = 0;
        ((acc[lane++] += static_cast<double>(ks(self))), ...);
    }

    for (int l = 0; l < NLanes; ++l) {
        sm[threadIdx.x] = acc[l];
        __syncthreads();
        for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sm[threadIdx.x] += sm[threadIdx.x + s];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            partials[(static_cast<long>(l) * gridDim.x) + blockIdx.x] = sm[0];
        }
        __syncthreads();
    }
}

// Σ_x k_i(φ(x)) for every kernel in the pack, one pass, returned as a
// std::tuple<double, ...> in kernel order — the same shape `obs::reduce`
// returns, so the `obs::*_of` finalizers consume it unchanged.
// Deliberately NOT offering a `reduce(f, stream, ks...)` sibling: the kernel
// pack would happily swallow a cudaStream_t as its first kernel, leaving the two
// overloads ambiguous for an exact-match stream argument. Measurement runs after
// `hmc.sync()` on the default stream; if a stream form is ever needed it wants a
// distinct name, not an overload.
template <class T, class Layout, class... Ks>
[[nodiscard]] inline auto reduce(DeviceField<T, Layout> const& f, Ks const&... ks) {
    static_assert(sizeof...(Ks) > 0, "cuda::reduce needs at least one kernel");
    constexpr int n_lanes     = sizeof...(Ks);
    cudaStream_t const stream = nullptr;

    auto const n = static_cast<long>(flat_size(f));
    std::vector<double> host(static_cast<std::size_t>(n_lanes) * k_obs_max_grid, 0.0);
    int grid = 0;
    if (n > 0) {
        grid = obs_grid_for(n);
        DeviceBuffer<double> partials{static_cast<std::size_t>(n_lanes) * grid};
        obs_reduce_kernel<n_lanes>
            <<<grid, k_obs_block, 0, stream>>>(f.data(), n, partials.data(), ks...);
        RETICOLO_CUDA_CHECK_LAUNCH();
        partials.copy_to_host(host.data(), stream);
        RETICOLO_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    auto lane_sum = [&](std::size_t lane) {
        double s = 0.0;
        for (int b = 0; b < grid; ++b) {  // index order — reproducible
            s += host[(lane * static_cast<std::size_t>(grid)) + static_cast<std::size_t>(b)];
        }
        return s;
    };
    return [&]<std::size_t... I>(std::index_sequence<I...>) {
        return std::make_tuple(lane_sum(I)...);
    }(std::index_sequence_for<Ks...>{});
}

}  // namespace reticolo::cuda
