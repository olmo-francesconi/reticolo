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

#include <cuda_runtime.h>

#include <cstddef>

namespace reticolo::cuda {

template <class HostAction, class Field>
class DeviceAction {
public:
    using traits = device_functors<HostAction>;

    DeviceAction(HostAction host, DeviceTopology topo, cudaStream_t stream = nullptr)
        : host_{host},
          topo_{topo},
          scratch_{static_cast<std::size_t>(topo.nsites)},
          stream_{stream} {}

    [[nodiscard]] double s_full(Field const& field) const {
        return reduce_fwd_launch(traits::make_energy(host_), field.data(), scratch_.data(), topo_,
                                 stream_);
    }

    void compute_force(Field const& field, Field& force) const {
        stencil_launch(traits::make_force(host_), field.data(), force.data(), topo_, stream_);
    }

private:
    HostAction host_;
    DeviceTopology topo_;
    mutable DeviceBuffer<double> scratch_;
    cudaStream_t stream_;
};

}  // namespace reticolo::cuda
