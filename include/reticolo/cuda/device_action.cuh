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
#include <cstdint>

#include <cuda_runtime.h>

namespace reticolo::cuda {

// The trait contract DeviceAction relies on: a device_functors<HostAction> with
// the four static launchers (force / s_full / s_full_into / sample_momenta), for
// the field element type T. A missing or mis-shaped (e.g. gauge) trait fails
// here with a readable message rather than deep inside a kernel launch.
template <class HostAction, class T>
concept DeviceActionTraits = requires(double* out,
                                      HostAction const& a,
                                      T const* field,
                                      T* force,
                                      double* scratch,
                                      double* partials,
                                      DeviceTopology const& topo,
                                      cudaStream_t s,
                                      std::uint64_t seed,
                                      std::uint64_t const* traj,
                                      long n) {
    device_functors<HostAction>::compute_force(a, field, force, topo, s);
    { device_functors<HostAction>::s_full(a, field, scratch, topo, s) } -> std::same_as<double>;
    device_functors<HostAction>::s_full_into(out, a, field, scratch, partials, topo, s);
    device_functors<HostAction>::sample_momenta(force, n, topo, seed, traj, s);
};

// Optional fused force+action launcher (site actions only). When present,
// DeviceAction exposes s_full_and_force and the LLR WindowedAction uses it in
// place of a separate compute_force + s_full_into. Absent → the two-pass path.
template <class HostAction, class T>
concept HasFusedForce = requires(double* out,
                                 HostAction const& a,
                                 T const* field,
                                 T* force,
                                 double* scratch,
                                 double* partials,
                                 DeviceTopology const& topo,
                                 cudaStream_t s) {
    device_functors<HostAction>::s_full_and_force(out, a, field, force, scratch, partials, topo, s);
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

    // Fused force + total action in one field gather: writes the force into
    // `force` and Σ S_base into out[0], no host sync. Present only when the trait
    // provides it (site actions with a force/energy functor); the LLR
    // WindowedAction detects it via `requires`.
    void s_full_and_force(double* out,
                          Field const& field,
                          Field& force,
                          double* partials,
                          cudaStream_t stream) const
        requires HasFusedForce<HostAction, typename Field::value_type>
    {
        traits::s_full_and_force(
            out, host_, field.data(), force.data(), scratch_.data(), partials, topo_, stream);
    }

    // Fill `mom` with fresh momenta — iid normals for scalar/U(1), a Gell-Mann
    // algebra draw for gauge groups. `n` is the momentum buffer's element count.
    void sample_momenta(typename Field::value_type* mom,
                        long n,
                        std::uint64_t seed,
                        std::uint64_t const* traj,
                        cudaStream_t stream) const {
        traits::sample_momenta(mom, n, topo_, seed, traj, stream);
    }

private:
    HostAction host_;
    DeviceTopology topo_;
    mutable DeviceBuffer<double> scratch_;
};

}  // namespace reticolo::cuda
