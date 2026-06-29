#pragma once

#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/reduce.hpp>
#include <reticolo/cuda/reduce_fwd.cuh>
#include <reticolo/cuda/stencil.cuh>

#include <cuda_runtime.h>

// The site-stencil access pattern shared by every scalar device action's
// device_functors specialization: force via the 2d-neighbour stencil, total
// action via the forward reduction. A scalar action supplies its force/energy
// functor; these thin wrappers launch the generic skeletons. The gauge actions
// use their own (gauge_u1.cuh) launchers instead — DeviceAction delegates to the
// trait, so the two access patterns never meet.

namespace reticolo::cuda::detail {

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

}  // namespace reticolo::cuda::detail
