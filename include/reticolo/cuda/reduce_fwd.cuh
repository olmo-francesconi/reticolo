#pragma once

// Generic forward-reduction skeleton — nvcc-only (.cuh; included from .cu).
//
// Companion to stencil.cuh for the total action `s_full`. Thread-per-site
// streams only the d FORWARD neighbours through f.accumulate(mu, nbr) (each
// bond counted once — the positive-mu convention of the CPU reduce_fwd), then
// writes the per-site contribution f.finalize() to a double scratch buffer.
// The volume sum is the deterministic reduce_sum_f64 (fixed launch config), so
// the result is bit-reproducible run-to-run — required because HMC
// reversibility depends on ΔH being computed the same way every trajectory.
//
// Splitting per-site-eval (here) from the reduction (reduce_sum_f64) keeps the
// summation order independent of the field layout and lets Phase 2 fuse
// s_full + kinetic over the same primitive.

#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/reduce.hpp>

#include <cuda_runtime.h>

namespace reticolo::cuda {

template <class F>
__global__ void reduce_fwd_site_kernel(F f, typename F::element const* field, double* site_out,
                                       DeviceTopology topo) {
    long const i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= topo.nsites) {
        return;
    }
    f.init(field[i]);
    for (int mu = 0; mu < topo.ndim; ++mu) {
        f.accumulate(mu, field[topo.next(i, mu)]);
    }
    site_out[i] = static_cast<double>(f.finalize());
}

// Σ_i F(field, i) over forward neighbours, in deterministic f64 order.
// `site_scratch` is a device buffer of at least topo.nsites doubles. Returns a
// host double (reduce_sum_f64 finishes on the host).
template <class F>
[[nodiscard]] double reduce_fwd_launch(F const& f, typename F::element const* field,
                                       double* site_scratch, DeviceTopology const& topo,
                                       cudaStream_t stream = nullptr) {
    constexpr int kBlock = 256;
    auto const grid = static_cast<unsigned>((topo.nsites + kBlock - 1) / kBlock);
    reduce_fwd_site_kernel<F><<<grid, kBlock, 0, stream>>>(f, field, site_scratch, topo);
    RETICOLO_CUDA_CHECK_LAUNCH();
    return reduce_sum_f64(site_scratch, topo.nsites, stream);
}

}  // namespace reticolo::cuda
