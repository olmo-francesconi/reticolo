#pragma once

#include <reticolo/action/site/bose_gas.hpp>
#include <reticolo/action/detail/site/bose_gas_formula.hpp>
#include <reticolo/core/cplx.hpp>
#include <reticolo/cuda/actions/device_functors.hpp>
#include <reticolo/cuda/actions/site_launchers.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/macros.hpp>
#include <reticolo/cuda/rng_philox.cuh>

#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>

// Device functors for the relativistic Bose gas (complex scalar). HMC samples the
// phase-quenched real part S_R only — its per-site math is the shared HD formula
// (bose_gas_formula.hpp), one source of truth with the CPU action::BoseGas. The
// field element is cplx<T> (AoS), flat-copy-compatible with the host
// Lattice<std::complex<T>>. Templated on T ∈ {double, float}.
//
// The time direction (mu == ndim-1) carries cosh(μ) on the hopping; the functor
// folds that weight in per accumulated neighbour, so the formula sees the already-
// weighted sum — exactly the CPU convention.

namespace reticolo::cuda {

// The complex-field HMC atoms (drift / kick / kinetic over the 2·n underlying
// reals) live in integ_ops.hpp so they are visible at the cuda::Hmc definition.

// --- per-site functors ---------------------------------------------------
template <class T>
class BoseGasForceFunctor {
public:
    using element = cplx<T>;
    RETICOLO_HD BoseGasForceFunctor(T coef_mass, T lambda, T ch, int last_dir)
        : coef_mass_{coef_mass}, lambda_{lambda}, ch_{ch}, last_{last_dir} {}

    RETICOLO_HD void init(cplx<T> self) {
        phi_    = self;
        staple_ = cplx<T>{};
    }
    RETICOLO_HD void accumulate(int mu, cplx<T> nbr) {
        staple_ += (mu == last_) ? (ch_ * nbr) : nbr;  // time dir ×cosh(μ)
    }
    [[nodiscard]] RETICOLO_HD cplx<T> finalize() const {
        return action::detail::bose_gas_force_site<T>(phi_, staple_, coef_mass_, lambda_);
    }

private:
    T coef_mass_;
    T lambda_;
    T ch_;
    int last_;
    cplx<T> phi_{};
    cplx<T> staple_{};
};

template <class T>
class BoseGasEnergyFunctor {
public:
    using element = cplx<T>;
    RETICOLO_HD BoseGasEnergyFunctor(T coef_mass, T lambda, T ch, int last_dir)
        : coef_mass_{coef_mass}, lambda_{lambda}, ch_{ch}, last_{last_dir} {}

    RETICOLO_HD void init(cplx<T> self) {
        phi_ = self;
        fwd_ = cplx<T>{};
    }
    RETICOLO_HD void accumulate(int mu, cplx<T> nbr) {
        fwd_ += (mu == last_) ? (ch_ * nbr) : nbr;  // forward only (reduce_fwd)
    }
    [[nodiscard]] RETICOLO_HD double finalize() const {
        return static_cast<double>(
            action::detail::bose_gas_action_site<T>(phi_, fwd_, coef_mass_, lambda_));
    }

private:
    T coef_mass_;
    T lambda_;
    T ch_;
    int last_;
    cplx<T> phi_{};
    cplx<T> fwd_{};
};

// --- trait ---------------------------------------------------------------
template <class T>
struct device_functors<action::BoseGas<T>> {
    [[nodiscard]] static T coef_mass(action::BoseGas<T> const& a, DeviceTopology const& topo) {
        return (T{2} * static_cast<T>(topo.ndim)) + (a.mass * a.mass);
    }
    [[nodiscard]] static T ch(action::BoseGas<T> const& a) {
        return static_cast<T>(std::cosh(static_cast<double>(a.mu)));
    }

    static void compute_force(action::BoseGas<T> const& a,
                              cplx<T> const* field,
                              cplx<T>* force,
                              DeviceTopology const& topo,
                              cudaStream_t s) {
        detail::site_force(
            BoseGasForceFunctor<T>{coef_mass(a, topo), a.lambda, ch(a), topo.ndim - 1},
            field,
            force,
            topo,
            s);
    }

    [[nodiscard]] static double s_full(action::BoseGas<T> const& a,
                                       cplx<T> const* field,
                                       double* scratch,
                                       DeviceTopology const& topo,
                                       cudaStream_t s) {
        return detail::site_s_full(
            BoseGasEnergyFunctor<T>{coef_mass(a, topo), a.lambda, ch(a), topo.ndim - 1},
            field,
            scratch,
            topo,
            s);
    }

    static void s_full_into(double* out,
                            action::BoseGas<T> const& a,
                            cplx<T> const* field,
                            double* scratch,
                            double* partials,
                            DeviceTopology const& topo,
                            cudaStream_t s) {
        detail::site_s_full_into(
            out,
            BoseGasEnergyFunctor<T>{coef_mass(a, topo), a.lambda, ch(a), topo.ndim - 1},
            field,
            scratch,
            partials,
            topo,
            s);
    }

    // Complex momentum: 2 iid normals per site = 2·n reals over the cplx buffer.
    static void sample_momenta(cplx<T>* mom,
                               long n,
                               DeviceTopology const& /*topo*/,
                               std::uint64_t seed,
                               std::uint64_t const* traj,
                               cudaStream_t s) {
        fill_normals(reinterpret_cast<T*>(mom), 2 * n, seed, traj, s);
    }
};

}  // namespace reticolo::cuda
