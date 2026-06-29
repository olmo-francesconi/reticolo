#pragma once

// Generic device action — nvcc-only (.cuh; launches kernels).
//
// Wraps a host action (couplings) and launches the stencil / reduce_fwd
// skeletons with the functor pair mapped by cuda::device_functors. It exposes
// the same s_full / compute_force(field, force) protocol the CPU HmcAction
// concept uses, so cuda::Hmc reuses the unchanged integrator atoms. A new
// device-ported action needs only a device_functors specialization — this
// wrapper never changes.

#include <reticolo/cuda/actions/phi4.hpp>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/reduce_fwd.cuh>
#include <reticolo/cuda/stencil.cuh>
#include <reticolo/cuda/stream.hpp>

#include <cuda_runtime.h>

#include <cstddef>

namespace reticolo::cuda {

template <class HostAction, class Field>
class DeviceAction {
public:
    using traits = device_functors<HostAction>;

    DeviceAction(HostAction host, DeviceTopology topo)
        : host_{host}, topo_{topo}, scratch_{static_cast<std::size_t>(topo.nsites)} {}

    // Launches on the thread-local current stream (cuda/stream.hpp), so a
    // compute_force inside a captured MD trajectory lands on the capture stream.
    [[nodiscard]] double s_full(Field const& field) const {
        return reduce_fwd_launch(traits::make_energy(host_), field.data(), scratch_.data(), topo_,
                                 current_stream());
    }

    void compute_force(Field const& field, Field& force) const {
        stencil_launch(traits::make_force(host_), field.data(), force.data(), topo_,
                       current_stream());
    }

    // Device-scalar s_full for the hot loop: writes the action to out[0] with no
    // host sync / no allocation (`partials` is caller-owned, k_reduce_max_grid).
    void s_full_into(double* out, Field const& field, double* partials, cudaStream_t stream) const {
        reduce_fwd_into(out, traits::make_energy(host_), field.data(), scratch_.data(), partials,
                        topo_, stream);
    }

private:
    HostAction host_;
    DeviceTopology topo_;
    mutable DeviceBuffer<double> scratch_;
};

}  // namespace reticolo::cuda
