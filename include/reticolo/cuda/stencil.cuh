#pragma once

// Generic scalar STENCIL skeleton — nvcc-only (.cuh; included from .cu).
//
// One __global__ template, parameterized by an action functor F (see the
// protocol in test_functors.hpp). Thread-per-site: each thread copies the POD
// functor into registers, streams all 2d nearest neighbours through
// f.accumulate(mu, nbr), and writes f.finalize() to out[i]. This is the
// access policy for the MD force / fused kick — every bond seen twice
// (forward and backward), matching the CPU visit_nn convention.
//
// The functor is passed BY VALUE so each thread gets its own accumulator in
// registers; for phi4 it degenerates to a scalar add, for heavier actions the
// streaming form keeps the per-direction partials in registers rather than a
// spilled local array.

#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_topology.hpp>

#include <cuda_runtime.h>

namespace reticolo::cuda {

// `field`/`out` are __restrict__ (the force write never aliases the field read)
// and `field` is const → lets nvcc route the 2d gather through the read-only data
// cache (LDG). Generic across every scalar element type (double/float/cplx<T>),
// unlike an explicit __ldg which lacks a complex overload. Measured neutral on
// P100 — the phi4 force is latency-bound (strided 4D gather, ~6% of peak BW),
// not redundant-read-bound, so the cache hint buys ~0. Kept as correct intent;
// a real phi4 win needs shared-memory halo tiling (deferred).
template <class F>
__global__ void stencil_kernel(F f, typename F::element const* __restrict__ field,
                               typename F::element* __restrict__ out, DeviceTopology topo) {
    long const i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= topo.nsites) {
        return;
    }
    f.init(field[i]);
    for (int mu = 0; mu < topo.ndim; ++mu) {
        f.accumulate(mu, field[topo.next(i, mu)]);
        f.accumulate(mu, field[topo.prev(i, mu)]);
    }
    out[i] = f.finalize();
}

// out[i] = F(field, i) over all 2d neighbours. Pointers are device pointers.
template <class F>
void stencil_launch(F const& f, typename F::element const* field, typename F::element* out,
                    DeviceTopology const& topo, cudaStream_t stream = nullptr) {
    constexpr int kBlock = 256;
    auto const grid = static_cast<unsigned>((topo.nsites + kBlock - 1) / kBlock);
    stencil_kernel<F><<<grid, kBlock, 0, stream>>>(f, field, out, topo);
    RETICOLO_CUDA_CHECK_LAUNCH();
}

}  // namespace reticolo::cuda
