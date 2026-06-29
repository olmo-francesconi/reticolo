#pragma once

#include <reticolo/action/compact_u1.hpp>
#include <reticolo/cuda/actions/device_functors.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/gauge_u1.cuh>
#include <reticolo/cuda/reduce.hpp>

#include <cuda_runtime.h>

// device_functors specialization for compact U(1). Unlike the scalar traits
// (site stencil), this wraps the gauge plaquette skeletons (gauge_u1.cuh): the
// force is a per-link gather, the action a per-site forward-plane sum reduced
// over the volume. The interface is identical to the scalar traits, so
// cuda::DeviceAction consumes it the same way — the access pattern is the only
// thing that differs, and it lives entirely here.

namespace reticolo::cuda {

template <class T>
struct device_functors<action::CompactU1<T>> {
    static void compute_force(action::CompactU1<T> const& a,
                              T const* field,
                              T* force,
                              DeviceTopology const& topo,
                              cudaStream_t s) {
        plaq_force_launch(field, force, topo, static_cast<double>(a.beta), s);
    }

    // The per-site kernel already folds in beta·(1 - cos), so Σ_x scratch is the
    // full Wilson action — a plain reduce_sum, no constant fixup.
    [[nodiscard]] static double s_full(action::CompactU1<T> const& a,
                                       T const* field,
                                       double* scratch,
                                       DeviceTopology const& topo,
                                       cudaStream_t s) {
        plaq_energy_launch(field, scratch, topo, static_cast<double>(a.beta), s);
        return reduce_sum_f64(scratch, topo.nsites, s);
    }

    static void s_full_into(double* out,
                            action::CompactU1<T> const& a,
                            T const* field,
                            double* scratch,
                            double* partials,
                            DeviceTopology const& topo,
                            cudaStream_t s) {
        plaq_energy_launch(field, scratch, topo, static_cast<double>(a.beta), s);
        reduce_sum_into(out, scratch, topo.nsites, partials, s);
    }
};

}  // namespace reticolo::cuda
