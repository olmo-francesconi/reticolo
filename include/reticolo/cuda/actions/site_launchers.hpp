#pragma once

#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/reduce.cuh>
#include <reticolo/cuda/reduce_fwd.cuh>
#include <reticolo/cuda/rng_philox.cuh>
#include <reticolo/cuda/stencil.cuh>
#include <reticolo/cuda/stencil_fwd_fused.cuh>

#include <cstdint>

#include <cuda_runtime.h>

// The site-stencil access pattern shared by every scalar device action's
// device_functors specialization: force via the 2d-neighbour stencil, total
// action via the forward reduction. A scalar action supplies its force/energy
// functor; these thin wrappers launch the generic skeletons. The gauge actions
// use their own (gauge_u1.cuh) launchers instead — DeviceAction delegates to the
// trait, so the two access patterns never meet.

namespace reticolo::cuda::impl {

template <class ForceF, class T>
inline void
site_force(ForceF f, T const* field, T* force, DeviceTopology const& topo, cudaStream_t s) {
    stencil_launch(f, field, force, topo, s);
}

template <class EnergyF, class T>
[[nodiscard]] inline double site_s_full(
    EnergyF e, T const* field, double* scratch, DeviceTopology const& topo, cudaStream_t s) {
    return reduce_fwd_launch(e, field, scratch, topo, s);
}

template <class EnergyF, class T>
inline void site_s_full_into(double* out,
                             EnergyF e,
                             T const* field,
                             double* scratch,
                             double* partials,
                             DeviceTopology const& topo,
                             cudaStream_t s) {
    reduce_fwd_into(out, e, field, scratch, partials, topo, s);
}

// Fused force + total action in one field gather (the dual-output stencil). The
// action opts in by naming a force/energy functor with the fwd/bwd/force/energy
// protocol; used by the LLR WindowedAction, whose force scale needs S_base on
// every MD step. Actions without a fused functor keep the two-pass path.
template <class FusedF, class T>
inline void site_s_full_and_force(double* out,
                                  FusedF f,
                                  T const* field,
                                  T* force,
                                  double* scratch,
                                  double* partials,
                                  DeviceTopology const& topo,
                                  cudaStream_t s) {
    stencil_fwd_fused_launch(f, field, force, out, scratch, partials, topo, s);
}

// Scalar / U(1) momentum sampler: one iid normal per field component. The
// gauge actions override this with a Gell-Mann algebra draw; `topo` is unused
// here (the count `n` is the buffer length) but kept for a uniform signature.
template <class T>
inline void site_sample_momenta(T* mom,
                                long n,
                                DeviceTopology const& /*topo*/,
                                std::uint64_t seed,
                                std::uint64_t const* traj,
                                cudaStream_t s) {
    fill_normals(mom, n, seed, traj, s);
}

}  // namespace reticolo::cuda::impl
