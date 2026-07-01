#pragma once

// Dual-output STENCIL skeleton — nvcc-only (.cuh; included from .cu).
//
// The fused twin of stencil.cuh + reduce_fwd.cuh: ONE thread-per-site neighbour
// gather that emits BOTH the MD force (all 2d neighbours) and the per-site
// action contribution (d forward neighbours). The LLR windowed force needs
// S_base on every MD step, so without fusion each force eval pays a stencil pass
// AND a separate reduce_fwd pass over the same field. This folds them into a
// single gather — the reduce_fwd pass was ~21% of the LLR GPU time.
//
// The functor SPLITS the gather: fwd(nbr) sees the d forward neighbours (fed to
// both the force's full 2d sum and the energy's forward sum), bwd(nbr) sees the
// d backward ones (force only). Keeping a separate forward accumulator (a free
// register on device — the CPU s_full_and_force can't, so it halves the hopping
// weight instead) makes energy() call the SAME per-site formula on the SAME
// forward sum as reduce_fwd_site_kernel, so the fused S_base is bit-identical to
// s_full_into's — no reproducibility divergence, no new formula.
//
// The per-site energies are staged to a double scratch buffer and summed with
// the deterministic reduce_sum_into (as reduce_fwd_into does), keeping ΔH
// bit-reproducible trajectory-to-trajectory.

#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/reduce.cuh>

#include <cuda_runtime.h>

namespace reticolo::cuda {

// field/force/site_energy __restrict__ + const field → read-only-cache (LDG)
// gather, as in stencil_kernel / reduce_fwd_site_kernel.
template <class F>
__global__ void stencil_fwd_fused_kernel(F f,
                                         typename F::element const* __restrict__ field,
                                         typename F::element* __restrict__ force,
                                         double* __restrict__ site_energy,
                                         DeviceTopology topo) {
    long const i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= topo.nsites) {
        return;
    }
    f.init(field[i]);
    for (int mu = 0; mu < topo.ndim; ++mu) {
        f.fwd(field[topo.next(i, mu)]);
        f.bwd(field[topo.prev(i, mu)]);
    }
    force[i]       = f.force();
    site_energy[i] = f.energy();
}

// force[i] = f.force(field, i) over all 2d neighbours; out[0] = Σ_i f.energy(field, i)
// over forward neighbours, in deterministic f64 order — all from one field
// gather, no host sync. `site_scratch` is topo.nsites doubles, `partials` is
// k_reduce_max_grid doubles; both caller-owned.
template <class F>
void stencil_fwd_fused_launch(F const& f,
                              typename F::element const* field,
                              typename F::element* force,
                              double* out,
                              double* site_scratch,
                              double* partials,
                              DeviceTopology const& topo,
                              cudaStream_t stream = nullptr) {
    constexpr int kBlock = 256;
    auto const grid      = static_cast<unsigned>((topo.nsites + kBlock - 1) / kBlock);
    stencil_fwd_fused_kernel<F><<<grid, kBlock, 0, stream>>>(f, field, force, site_scratch, topo);
    RETICOLO_CUDA_CHECK_LAUNCH();
    reduce_sum_into(out, site_scratch, topo.nsites, partials, stream);
}

}  // namespace reticolo::cuda
