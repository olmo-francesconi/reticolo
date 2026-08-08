#pragma once

// Device-side fused observable reduction — the GPU twin of `obs::reduce` /
// `obs::reduce_nn` (obs/reduce.hpp), with the SAME contract on both backends:
//
//   reduce(field, ks...)             lane i = Σ_x k_i(φ(x))
//   reduce_nn<Policy>(field, ks...)  lane i = Σ_x k_i(φ(x), agg(x))
//
// A PACK of per-site kernels runs in ONE pass over a `DeviceField`, so N
// observables cost one read of VRAM instead of N. Lanes are heterogeneous: each
// accumulates in its own kernel's return type, exactly like the host `Lanes`
// tuple, and both entry points return a std::tuple of the raw sums. `Policy` is
// the SAME `exec::FwdOnly` / `exec::AllDirs` tag the CPU stencil takes, so the
// neighbour aggregate means the same thing on both sides.
//
// Dispatch is by ADL on the field type — the seam the integrator atoms already
// use (`drift_field`/`kick_add`). `obs::reduce` on a host `Lattice` and
// `cuda::reduce` on a `DeviceField` are SEPARATE overloads in separate
// namespaces; there is deliberately no dispatcher inside `obs::` that names
// `reduce` with a dependent argument, because such a body would bind names
// differently in a .cu TU than in a .cpp TU (IFNDR — and both TU kinds link into
// every CUDA test binary). Call it unqualified, or as `cuda::reduce`.
//
// Kernels are the SAME objects the CPU path folds: `obs::kernel::*` are
// RETICOLO_HD, so `phi_sq` here and `phi_sq` there are one definition.
//
// Heterogeneous lanes work by decomposing each lane into its double components
// (a real lane is 1, a `cplx` lane is 2), reducing the flattened component
// array, and reassembling on the host — ONE block-reduction kernel serves every
// lane shape instead of a per-type kernel family.
//
// Determinism: the fixed launch config from reduce.cuh (kBlock/kMaxGrid — one
// configuration for the whole backend), a grid-stride loop, per-component block
// partials, and a host finish folding partials in index order — reproducible
// run-to-run. NOT bit-identical to the CPU fold (different tree): the project's
// standing CPU↔GPU position is same ensemble, not same chain.
//
// No `reduce(f, stream, ks...)` sibling: a kernel pack would swallow a
// cudaStream_t as its first kernel, leaving the overloads ambiguous. Measurement
// runs after `hmc.sync()` on the default stream.

#include <reticolo/core/exec/nn_stencil.hpp>
#include <reticolo/core/field/cplx.hpp>
#include <reticolo/core/sys/hd.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/reduce.cuh>
#include <reticolo/obs/concepts.hpp>

#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace reticolo::cuda {

// ---- lane <-> double-component mapping --------------------------------------

template <class R>
struct lane_traits {
    static constexpr int width = 1;
    RETICOLO_HD static void add(double* dst, R v) { dst[0] += static_cast<double>(v); }
    [[nodiscard]] static R load(double const* src) { return static_cast<R>(src[0]); }
};

template <class T>
struct lane_traits<cplx<T>> {
    static constexpr int width = 2;
    RETICOLO_HD static void add(double* dst, cplx<T> v) {
        dst[0] += static_cast<double>(v.re);
        dst[1] += static_cast<double>(v.im);
    }
    [[nodiscard]] static cplx<T> load(double const* src) {
        return cplx<T>{static_cast<T>(src[0]), static_cast<T>(src[1])};
    }
};

// Total double components across a lane pack — the flattened width the block
// reduction actually folds.
template <class... Rs>
inline constexpr int lane_total_width_v = (lane_traits<Rs>::width + ... + 0);

// Fold this block's per-component accumulators into
// partials[(c * gridDim.x) + blockIdx.x]. Shared scratch is reused component by
// component, so the shared footprint is one block of doubles regardless of lane
// count.
template <int Width>
RETICOLO_DEVICE inline void
obs_block_fold(double (&acc)[Width], double* sm, double* __restrict__ partials) {
    for (int c = 0; c < Width; ++c) {
        sm[threadIdx.x] = acc[c];
        __syncthreads();
        for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
            if (threadIdx.x < s) {
                sm[threadIdx.x] += sm[threadIdx.x + s];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            partials[(static_cast<long>(c) * gridDim.x) + blockIdx.x] = sm[0];
        }
        __syncthreads();
    }
}

// ---- kernels ----------------------------------------------------------------

template <int Width, class T, class... Ks>
__global__ void
obs_reduce_kernel(T const* __restrict__ x, long n, double* __restrict__ partials, Ks... ks) {
    __shared__ double sm[kBlock];  // NOLINT(cppcoreguidelines-avoid-c-arrays)

    double acc[Width] = {};  // NOLINT(cppcoreguidelines-avoid-c-arrays)
    for (long i = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x; i < n;
         i += static_cast<long>(gridDim.x) * blockDim.x) {
        T const self = x[i];
        int off      = 0;
        ((lane_traits<decltype(ks(self))>::add(acc + off, ks(self)),
          off += lane_traits<decltype(ks(self))>::width),
         ...);
    }
    obs_block_fold<Width>(acc, sm, partials);
}

// `agg` folds the Policy-selected neighbours (forward only for an action
// density, all 2d for a force functional) — the CPU `exec::nn_reduce` contract
// with `IdentityCombine`.
template <int Width, class Policy, class T, class... Ks>
__global__ void obs_reduce_nn_kernel(T const* __restrict__ x,
                                     double* __restrict__ partials,
                                     DeviceTopology topo,
                                     Ks... ks) {
    __shared__ double sm[kBlock];  // NOLINT(cppcoreguidelines-avoid-c-arrays)

    double acc[Width] = {};  // NOLINT(cppcoreguidelines-avoid-c-arrays)
    for (long i = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x; i < topo.nsites;
         i += static_cast<long>(gridDim.x) * blockDim.x) {
        T const self = x[i];
        T agg{};
        for (int mu = 0; mu < topo.ndim; ++mu) {
            if constexpr (!std::is_same_v<Policy, exec::BwdOnly>) {
                agg += x[topo.next(i, mu)];
            }
            if constexpr (!std::is_same_v<Policy, exec::FwdOnly>) {
                agg += x[topo.prev(i, mu)];
            }
        }
        int off = 0;
        ((lane_traits<decltype(ks(self, agg))>::add(acc + off, ks(self, agg)),
          off += lane_traits<decltype(ks(self, agg))>::width),
         ...);
    }
    obs_block_fold<Width>(acc, sm, partials);
}

// ---- host entry points ------------------------------------------------------

namespace impl {

// Fold grid partials for each lane in index order and reassemble the lane's own
// type from its double components.
template <class... Rs, std::size_t... I>
[[nodiscard]] inline std::tuple<Rs...>
assemble(double const* host, int grid, std::index_sequence<I...>) {
    constexpr int widths[]  = {lane_traits<Rs>::width...};  // NOLINT
    int offs[sizeof...(Rs)] = {};                           // NOLINT
    int run                 = 0;
    for (std::size_t k = 0; k < sizeof...(Rs); ++k) {
        offs[k] = run;
        run += widths[k];
    }
    auto lane = [&]<class R>(int off) {
        double comp[lane_traits<R>::width] = {};  // NOLINT
        for (int c = 0; c < lane_traits<R>::width; ++c) {
            double s = 0.0;
            for (int b = 0; b < grid; ++b) {
                s += host[(static_cast<long>(off + c) * grid) + b];
            }
            comp[c] = s;
        }
        return lane_traits<R>::load(comp);
    };
    return std::tuple<Rs...>{lane.template operator()<Rs>(offs[I])...};
}

}  // namespace impl

// Σ_x k_i(φ(x)) per lane, one pass. Returns std::tuple<R...> in kernel order —
// the same shape `obs::reduce` returns, so the `obs::*_of` finalizers apply
// unchanged.
template <class T, class Layout, class... Ks>
    requires(obs::SiteKernel<Ks, T> && ...)
[[nodiscard]] inline auto reduce(DeviceField<T, Layout> const& f, Ks const&... ks) {
    static_assert(sizeof...(Ks) > 0, "cuda::reduce needs at least one kernel");
    constexpr int width = lane_total_width_v<std::decay_t<std::invoke_result_t<Ks const&, T>>...>;

    auto const n = static_cast<long>(flat_size(f));
    std::vector<double> host(static_cast<std::size_t>(width) * kMaxGrid, 0.0);
    int grid = 0;
    if (n > 0) {
        grid = grid_for(n);
        DeviceBuffer<double> partials{static_cast<std::size_t>(width) * grid};
        obs_reduce_kernel<width><<<grid, kBlock>>>(f.data(), n, partials.data(), ks...);
        RETICOLO_CUDA_CHECK_LAUNCH();
        partials.copy_to_host(host.data());
        RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
    return impl::assemble<std::decay_t<std::invoke_result_t<Ks const&, T>>...>(
        host.data(), grid, std::index_sequence_for<Ks...>{});
}

// Σ_x k_i(φ(x), agg(x)) per lane, one pass. `Policy` selects the neighbour
// aggregate exactly as on the host.
template <class Policy = exec::FwdOnly, class T, class Layout, class... Ks>
    requires(obs::NnKernel<Ks, T> && ...)
[[nodiscard]] inline auto reduce_nn(DeviceField<T, Layout> const& f, Ks const&... ks) {
    static_assert(sizeof...(Ks) > 0, "cuda::reduce_nn needs at least one kernel");
    constexpr int width =
        lane_total_width_v<std::decay_t<std::invoke_result_t<Ks const&, T, T>>...>;

    auto const topo = f.topology();
    std::vector<double> host(static_cast<std::size_t>(width) * kMaxGrid, 0.0);
    int grid = 0;
    if (topo.nsites > 0) {
        grid = grid_for(topo.nsites);
        DeviceBuffer<double> partials{static_cast<std::size_t>(width) * grid};
        obs_reduce_nn_kernel<width, Policy>
            <<<grid, kBlock>>>(f.data(), partials.data(), topo, ks...);
        RETICOLO_CUDA_CHECK_LAUNCH();
        partials.copy_to_host(host.data());
        RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }
    return impl::assemble<std::decay_t<std::invoke_result_t<Ks const&, T, T>>...>(
        host.data(), grid, std::index_sequence_for<Ks...>{});
}

}  // namespace reticolo::cuda
