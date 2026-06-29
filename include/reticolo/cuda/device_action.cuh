#pragma once

// Generic device action — nvcc-only (.cuh; launches kernels).
//
// Wraps a host action (its couplings) and a topology, and exposes the
// s_full / s_full_into / compute_force(field, force) protocol that cuda::Hmc and
// the reused integrator atoms consume. The ACCESS PATTERN — scalar site stencil
// vs gauge per-link plaquette gather — lives entirely in the
// device_functors<HostAction> trait's static launchers; DeviceAction delegates
// to them and never branches. A new device-ported action needs only a
// device_functors specialization (site or gauge), this wrapper never changes.

#include <reticolo/cuda/actions/device_functors.hpp>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/stream.hpp>

#include <concepts>
#include <cstddef>

#include <cuda_runtime.h>

namespace reticolo::cuda {

// The trait contract DeviceAction relies on: a device_functors<HostAction> with
// the three static launchers, for the field element type T. A missing or
// mis-shaped (e.g. gauge) trait fails here with a readable message rather than
// deep inside a kernel launch.
template <class HostAction, class T>
concept DeviceActionTraits = requires(double* out,
                                      HostAction const& a,
                                      T const* field,
                                      T* force,
                                      double* scratch,
                                      double* partials,
                                      DeviceTopology const& topo,
                                      cudaStream_t s) {
    device_functors<HostAction>::compute_force(a, field, force, topo, s);
    { device_functors<HostAction>::s_full(a, field, scratch, topo, s) } -> std::same_as<double>;
    device_functors<HostAction>::s_full_into(out, a, field, scratch, partials, topo, s);
};

template <class HostAction, class Field>
    requires DeviceActionTraits<HostAction, typename Field::value_type>
class DeviceAction {
public:
    using traits = device_functors<HostAction>;

    DeviceAction(HostAction host, DeviceTopology topo)
        : host_{host}, topo_{topo}, scratch_{static_cast<std::size_t>(topo.nsites)} {}

    // Launches on the thread-local current stream (cuda/stream.hpp), so a
    // compute_force inside a captured MD trajectory lands on the capture stream.
    [[nodiscard]] double s_full(Field const& field) const {
        return traits::s_full(host_, field.data(), scratch_.data(), topo_, current_stream());
    }

    void compute_force(Field const& field, Field& force) const {
        traits::compute_force(host_, field.data(), force.data(), topo_, current_stream());
    }

    // Device-scalar s_full for the hot loop: writes the action to out[0] with no
    // host sync / no allocation (`partials` is caller-owned, k_reduce_max_grid).
    void s_full_into(double* out, Field const& field, double* partials, cudaStream_t stream) const {
        traits::s_full_into(out, host_, field.data(), scratch_.data(), partials, topo_, stream);
    }

private:
    HostAction host_;
    DeviceTopology topo_;
    mutable DeviceBuffer<double> scratch_;
};

}  // namespace reticolo::cuda
