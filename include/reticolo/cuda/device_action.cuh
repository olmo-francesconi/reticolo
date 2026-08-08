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

// Optional local-update launcher — the device peer of action::LocalAction, and
// the same layering: the ACTION drives one colour of a checkerboard sweep over
// the lattice, cuda::Metropolis only says when. Present → DeviceAction exposes
// metropolis_stencil and cuda::Metropolis instantiates; absent → the action is
// HMC-only on the device, exactly as on the host.
template <class HostAction, class T>
concept HasLocalUpdate = requires(HostAction const& a,
                                  T* field,
                                  DeviceTopology const& topo,
                                  int color,
                                  double sigma,
                                  std::uint64_t seed,
                                  std::uint64_t const* sweep,
                                  double* ds_out,
                                  double* acc_out,
                                  cudaStream_t s) {
    device_functors<HostAction>::metropolis_stencil(
        a, field, topo, color, sigma, seed, sweep, ds_out, acc_out, s);
    { device_functors<HostAction>::n_colors(topo) } -> std::convertible_to<int>;
};

// Optional imaginary-part launchers (complex actions only, e.g. BoseGas). When
// present, DeviceAction exposes s_imag / s_imag_into / compute_force_imag and the
// LLR WindowedAction selects mode B (sample S_R, constrain S_I). Absent → mode A.
template <class HostAction, class T>
concept HasImagDevice = requires(double* out,
                                 HostAction const& a,
                                 T const* field,
                                 T* force,
                                 double* scratch,
                                 double* partials,
                                 DeviceTopology const& topo,
                                 cudaStream_t s) {
    { device_functors<HostAction>::s_imag(a, field, scratch, topo, s) } -> std::same_as<double>;
    device_functors<HostAction>::s_imag_into(out, a, field, scratch, partials, topo, s);
    device_functors<HostAction>::compute_force_imag(a, field, force, topo, s);
};

// Optional fused F_I + S_I launcher — one field gather yields both the imaginary
// force and the S_I reduction. The mode-B WindowedAction uses it when present;
// absent → the two-pass (compute_force_imag + s_imag_into).
template <class HostAction, class T>
concept HasFusedImagDevice = requires(double* out,
                                      HostAction const& a,
                                      T const* field,
                                      T* force,
                                      double* scratch,
                                      double* partials,
                                      DeviceTopology const& topo,
                                      cudaStream_t s) {
    device_functors<HostAction>::force_imag_and_s_imag_into(
        out, a, field, force, scratch, partials, topo, s);
};

template <class HostAction, class Field>
    requires DeviceActionTraits<HostAction, typename Field::value_type>
class DeviceAction {
public:
    using traits = device_functors<HostAction>;

    // scratch_ holds per-site energy for site actions (nsites) but also the
    // per-LINK energy partials of the fused gauge path (ndim·nsites, one plaquette
    // owner per link), so size it for the larger. Negligible extra device memory.
    DeviceAction(HostAction host, DeviceTopology topo)
        : host_{host}, topo_{topo},
          scratch_{static_cast<std::size_t>(topo.ndim) * static_cast<std::size_t>(topo.nsites)} {}

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
    void s_full_and_force(
        double* out, Field const& field, Field& force, double* partials, cudaStream_t stream) const
        requires HasFusedForce<HostAction, typename Field::value_type>
    {
        traits::s_full_and_force(
            out, host_, field.data(), force.data(), scratch_.data(), partials, topo_, stream);
    }

    // One colour of a local Metropolis sweep: updates `field` in place and stages
    // the per-site applied ΔS / accept flag for the caller's reduction. The host
    // twin is action::LocalAction::metropolis_stencil; the difference is only where
    // the randomness comes from — Philox is counter-based, so the device draws it
    // in-kernel from (seed, sweep counter, site) rather than reading pre-filled
    // noise fields, matching how sample_momenta already works on this side.
    void metropolis_stencil(Field& field,
                            int color,
                            double sigma,
                            std::uint64_t seed,
                            std::uint64_t const* sweep,
                            double* ds_out,
                            double* acc_out,
                            cudaStream_t stream) const
        requires HasLocalUpdate<HostAction, typename Field::value_type>
    {
        traits::metropolis_stencil(
            host_, field.data(), topo_, color, sigma, seed, sweep, ds_out, acc_out, stream);
    }

    // How many colour passes make one sweep: 2 for a site field, 2*ndim for a
    // link field, whose elementary move is per (direction, site parity).
    [[nodiscard]] int n_colors() const
        requires HasLocalUpdate<HostAction, typename Field::value_type>
    {
        return traits::n_colors(topo_);
    }

    // The site count the per-site staging buffers must hold — NOT field.size(),
    // which for a link field counts every component of every direction.
    [[nodiscard]] long nsites() const { return topo_.nsites; }

    // --- imaginary part (complex actions; instantiate only when the trait has it) ---
    [[nodiscard]] double s_imag(Field const& field) const
        requires HasImagDevice<HostAction, typename Field::value_type>
    {
        return traits::s_imag(host_, field.data(), scratch_.data(), topo_, current_stream());
    }

    void s_imag_into(double* out, Field const& field, double* partials, cudaStream_t stream) const
        requires HasImagDevice<HostAction, typename Field::value_type>
    {
        traits::s_imag_into(out, host_, field.data(), scratch_.data(), partials, topo_, stream);
    }

    void compute_force_imag(Field const& field, Field& force) const
        requires HasImagDevice<HostAction, typename Field::value_type>
    {
        traits::compute_force_imag(host_, field.data(), force.data(), topo_, current_stream());
    }

    // Fused: F_I into `force` and S_I into out[0], one field gather.
    void compute_force_imag_and_s_imag_into(
        double* out, Field const& field, Field& force, double* partials, cudaStream_t stream) const
        requires HasFusedImagDevice<HostAction, typename Field::value_type>
    {
        traits::force_imag_and_s_imag_into(
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
