#pragma once

#include <reticolo/action/complex/bose_gas.hpp>
#include <reticolo/action/complex/formula/bose_gas_formula.hpp>
#include <reticolo/core/cplx.hpp>
#include <reticolo/cuda/actions/device_functors.hpp>
#include <reticolo/cuda/actions/site_launchers.hpp>
#include <reticolo/cuda/bose_imag.cuh>
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

    // --- imaginary part (complex-LLR / phase-quenched S_I tracking) ----------
    // S_I / F_I are μ-independent and touch only the time direction; they call
    // the shared bose_gas_*_imag_site formula (bose_imag.cuh). Presence of these
    // two members is what makes the device LLR WindowedAction pick mode B.

    // S_I to a host double — used by the phase-quenched HMC app to append S_I to
    // its output Series (no window involved).
    [[nodiscard]] static double s_imag(action::BoseGas<T> const& /*a*/,
                                       cplx<T> const* field,
                                       double* scratch,
                                       DeviceTopology const& topo,
                                       cudaStream_t s) {
        return bose_s_imag_launch<T>(field, scratch, topo, s);
    }

    // Device-scalar S_I for the LLR hot loop (windowed action + constraint).
    static void s_imag_into(double* out,
                            action::BoseGas<T> const& /*a*/,
                            cplx<T> const* field,
                            double* scratch,
                            double* partials,
                            DeviceTopology const& topo,
                            cudaStream_t s) {
        bose_s_imag_into<T>(out, field, scratch, partials, topo, s);
    }

    static void compute_force_imag(action::BoseGas<T> const& /*a*/,
                                   cplx<T> const* field,
                                   cplx<T>* force,
                                   DeviceTopology const& topo,
                                   cudaStream_t s) {
        bose_force_imag_launch<T>(field, force, topo, s);
    }

    // Fused F_I + S_I in one τ-sweep — the LLR WindowedAction's mode-B force uses
    // this (drops the separate S_I reduction the two-pass path pays per MD step).
    static void force_imag_and_s_imag_into(double* out,
                                           action::BoseGas<T> const& /*a*/,
                                           cplx<T> const* field,
                                           cplx<T>* force,
                                           double* scratch,
                                           double* partials,
                                           DeviceTopology const& topo,
                                           cudaStream_t s) {
        bose_force_imag_and_s_imag_into<T>(out, field, force, scratch, partials, topo, s);
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

    // LLR hot-start: disorder φ ~ N(0, sigma²) over the 2·n underlying reals
    // before warm-in — the complex twin of the gauge hot_start, mirroring the
    // CPU bose_gas_llr hot_start. `n` is the cplx-buffer length (nsites).
    static void hot_start(cplx<T>* field,
                          long n,
                          double sigma,
                          std::uint64_t seed,
                          std::uint64_t const* traj,
                          cudaStream_t s) {
        fill_normals(reinterpret_cast<T*>(field), 2 * n, seed, traj, s, sigma);
    }
};

}  // namespace reticolo::cuda
