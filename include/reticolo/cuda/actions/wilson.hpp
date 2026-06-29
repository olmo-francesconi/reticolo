#pragma once

#include <reticolo/action/wilson.hpp>
#include <reticolo/cuda/actions/device_functors.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/gauge/group_device.hpp>
#include <reticolo/cuda/gauge_sun.cuh>
#include <reticolo/cuda/reduce.hpp>

#include <cstdint>

#include <cuda_runtime.h>

// device_functors specialization for the Wilson SU(N) gauge action. Wraps the
// generic gauge kernels (gauge_sun.cuh) through GD = group_device<G>::type —
// the device matrix-ops traits for the group. Same trait interface as the
// scalar / U(1) actions (force / s_full / s_full_into / sample_momenta), so
// cuda::DeviceAction consumes it identically; the per-link gather force, the
// plaquette action, and the Gell-Mann momentum sampler live in the kernels.
// SU(N) links are f64 only — T is double.

namespace reticolo::cuda {

template <class G, class T>
struct device_functors<action::Wilson<G, T>> {
    using GD = typename group_device<G>::type;

    static void compute_force(action::Wilson<G, T> const& a,
                              T const* field,
                              T* force,
                              DeviceTopology const& topo,
                              cudaStream_t s) {
        double const scale = -static_cast<double>(a.beta) / static_cast<double>(GD::n_color);
        su_plaq_force_launch<GD>(field, force, topo, scale, s);
    }

    [[nodiscard]] static double s_full(action::Wilson<G, T> const& a,
                                       T const* field,
                                       double* scratch,
                                       DeviceTopology const& topo,
                                       cudaStream_t s) {
        su_plaq_energy_launch<GD>(field, scratch, topo, static_cast<double>(a.beta), s);
        return reduce_sum_f64(scratch, topo.nsites, s);
    }

    static void s_full_into(double* out,
                            action::Wilson<G, T> const& a,
                            T const* field,
                            double* scratch,
                            double* partials,
                            DeviceTopology const& topo,
                            cudaStream_t s) {
        su_plaq_energy_launch<GD>(field, scratch, topo, static_cast<double>(a.beta), s);
        reduce_sum_into(out, scratch, topo.nsites, partials, s);
    }

    static void sample_momenta(T* mom,
                               long /*n*/,
                               DeviceTopology const& topo,
                               std::uint64_t seed,
                               std::uint64_t const* traj,
                               cudaStream_t s) {
        su_sample_algebra_launch<GD>(mom, topo, seed, traj, s);
    }
};

}  // namespace reticolo::cuda
