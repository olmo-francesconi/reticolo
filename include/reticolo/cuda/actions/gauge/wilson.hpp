#pragma once

#include <reticolo/action/gauge/wilson.hpp>
#include <reticolo/cuda/actions/device_functors.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/gauge/gauge_sun.cuh>
#include <reticolo/cuda/gauge/gauge_u1.cuh>
#include <reticolo/cuda/gauge/group_device.hpp>
#include <reticolo/cuda/reduce.cuh>
#include <reticolo/cuda/rng_philox.cuh>
#include <reticolo/math/group/u1.hpp>

#include <cstdint>

#include <cuda_runtime.h>

// device_functors specialization for the Wilson SU(N) gauge action. Wraps the
// generic gauge kernels (gauge_sun.cuh) through GD = group_device<G>::type —
// the device matrix-ops traits for the group. Same trait interface as the
// scalar / U(1) actions (force / s_full / s_full_into / sample_momenta), so
// cuda::DeviceAction consumes it identically; the per-link gather force, the
// plaquette action, and the Gell-Mann momentum sampler live in the kernels.
// SU(N) links are f64 only — T is double.
//
// U(1) is abelian and does NOT use the matrix path: the Wilson<U1> specialization
// below reuses the dedicated angle kernels (gauge_u1.cuh) on a 1-angle link,
// identical to CompactU1 (n_color=1 ⇒ β/N = β). Keeping the abelian kernels
// separate from gauge_sun.cuh is deliberate — a future non-SU(N) family (e.g. a
// symplectic group) plugs in as its own kernel set + functor, not a contortion of
// the generic matrix path.

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

    // Valid cold start = every link the group identity (the zero the default
    // buffer holds is not a group element). Detected by Replica::cold_start; the
    // abelian U(1) omits this (θ = 0 is already identity, memset suffices).
    static void set_cold(T* field, DeviceTopology const& topo, cudaStream_t s) {
        su_set_identity_launch<GD>(field, topo, s);
    }

    // Fused force + action in one staple gather (skips the redundant plaquette
    // pass) — used by the LLR WindowedAction. `scratch` is the ndim·nsites
    // per-link energy partials.
    static void s_full_and_force(double* out,
                                 action::Wilson<G, T> const& a,
                                 T const* field,
                                 T* force,
                                 double* scratch,
                                 double* partials,
                                 DeviceTopology const& topo,
                                 cudaStream_t s) {
        su_plaq_fused_launch<GD>(field, force, scratch, topo, static_cast<double>(a.beta), s);
        reduce_sum_into(out, scratch, topo.nsites * topo.ndim, partials, s);
    }
};

// Wilson<U(1)> — abelian specialization (more specialized than Wilson<G> above,
// so it wins for G = U1). Reuses the dedicated angle kernels (gauge_u1.cuh) on a
// 1-angle LinkLayout link, NOT the matrix path: a U(1) link is a phase, the drift
// is additive (θ ← θ + dt·p, the generic axpy atom), and the momentum is one iid
// normal per link. Bodies mirror device_functors<CompactU1> (n_color = 1).
template <class T>
struct device_functors<action::Wilson<math::group::U1, T>> {
    static void compute_force(action::Wilson<math::group::U1, T> const& a,
                              T const* field,
                              T* force,
                              DeviceTopology const& topo,
                              cudaStream_t s) {
        plaq_force_launch(field, force, topo, static_cast<double>(a.beta), s);
    }

    [[nodiscard]] static double s_full(action::Wilson<math::group::U1, T> const& a,
                                       T const* field,
                                       double* scratch,
                                       DeviceTopology const& topo,
                                       cudaStream_t s) {
        plaq_energy_launch(field, scratch, topo, static_cast<double>(a.beta), s);
        return reduce_sum_f64(scratch, topo.nsites, s);
    }

    static void s_full_into(double* out,
                            action::Wilson<math::group::U1, T> const& a,
                            T const* field,
                            double* scratch,
                            double* partials,
                            DeviceTopology const& topo,
                            cudaStream_t s) {
        plaq_energy_launch(field, scratch, topo, static_cast<double>(a.beta), s);
        reduce_sum_into(out, scratch, topo.nsites, partials, s);
    }

    // Fused Σ Re Tr U_p + force in one sincos device pass — the gauge twin of the
    // site s_full_and_force, used by the LLR WindowedAction. `scratch` is the
    // ndim·nsites per-link energy partials.
    static void s_full_and_force(double* out,
                                 action::Wilson<math::group::U1, T> const& a,
                                 T const* field,
                                 T* force,
                                 double* scratch,
                                 double* partials,
                                 DeviceTopology const& topo,
                                 cudaStream_t s) {
        plaq_fused_launch(field, force, scratch, topo, static_cast<double>(a.beta), s);
        reduce_sum_into(out, scratch, topo.nsites * topo.ndim, partials, s);
    }

    static void sample_momenta(T* mom,
                               long n,
                               DeviceTopology const& /*topo*/,
                               std::uint64_t seed,
                               std::uint64_t const* traj,
                               cudaStream_t s) {
        fill_normals(mom, n, seed, traj, s);
    }
};

}  // namespace reticolo::cuda
