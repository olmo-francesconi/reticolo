#pragma once

#include <reticolo/action/nn/formula/phi4_formula.hpp>
#include <reticolo/action/nn/phi4.hpp>
#include <reticolo/cuda/actions/device_functors.hpp>
#include <reticolo/cuda/actions/site_launchers.hpp>
#include <reticolo/cuda/macros.hpp>

#include <cstdint>

// Device per-site functors for Phi4 + the host-action → device-functor trait.
//
// The functors follow the scalar device protocol: init(self) /
// accumulate(int mu, T nbr) / finalize(). Their per-site math is the SHARED HD
// formula in action::formula (phi4_formula.hpp) — one source of truth with the
// CPU action::Phi4. They are trivially copyable PODs passed by value into the
// kernels (each thread gets its own accumulator in registers).

namespace reticolo::cuda {

template <class T>
class Phi4ForceFunctor {
public:
    using element = T;
    RETICOLO_HD Phi4ForceFunctor(T kappa, T lambda) : kappa_{kappa}, lambda_{lambda} {}

    RETICOLO_HD void init(T self) {
        phi_  = self;
        nbrs_ = T{0};
    }
    RETICOLO_HD void accumulate(int /*mu*/, T nbr) { nbrs_ += nbr; }
    [[nodiscard]] RETICOLO_HD T finalize() const {
        return action::formula::phi4_force_site<T>(phi_, nbrs_, kappa_, lambda_);
    }

private:
    T kappa_;
    T lambda_;
    T phi_  = T{0};
    T nbrs_ = T{0};
};

template <class T>
class Phi4EnergyFunctor {
public:
    using element = T;
    RETICOLO_HD Phi4EnergyFunctor(T kappa, T lambda) : kappa_{kappa}, lambda_{lambda} {}

    RETICOLO_HD void init(T self) {
        phi_ = self;
        fwd_ = T{0};
    }
    RETICOLO_HD void accumulate(int /*mu*/, T nbr) { fwd_ += nbr; }
    [[nodiscard]] RETICOLO_HD double finalize() const {
        return static_cast<double>(
            action::formula::phi4_action_site<T>(phi_, fwd_, kappa_, lambda_));
    }

private:
    T kappa_;
    T lambda_;
    T phi_ = T{0};
    T fwd_ = T{0};
};

// Fused force + energy: one neighbour gather yields both. Keeps a separate
// forward accumulator (fwd_) alongside the full 2d sum (full_) so energy() reuses
// the unchanged phi4_action_site on the forward sum — bit-identical to the
// reduce_fwd path, so the LLR windowed force's S_base matches s_full_into exactly.
template <class T>
class Phi4ForceEnergyFunctor {
public:
    using element = T;
    RETICOLO_HD Phi4ForceEnergyFunctor(T kappa, T lambda) : kappa_{kappa}, lambda_{lambda} {}

    RETICOLO_HD void init(T self) {
        phi_  = self;
        full_ = T{0};
        fwd_  = T{0};
    }
    RETICOLO_HD void fwd(T nbr) {
        full_ += nbr;
        fwd_ += nbr;
    }
    RETICOLO_HD void bwd(T nbr) { full_ += nbr; }
    [[nodiscard]] RETICOLO_HD T force() const {
        return action::formula::phi4_force_site<T>(phi_, full_, kappa_, lambda_);
    }
    [[nodiscard]] RETICOLO_HD double energy() const {
        return static_cast<double>(
            action::formula::phi4_action_site<T>(phi_, fwd_, kappa_, lambda_));
    }

private:
    T kappa_;
    T lambda_;
    T phi_  = T{0};
    T full_ = T{0};
    T fwd_  = T{0};
};

// Local-update functor (checkerboard.cuh's init/accumulate/ds protocol). No new
// physics and no new formula file: the per-site energy of x is phi4_action_site
// evaluated on the FULL 2d neighbour sum — each bond touching x counted once —
// so ΔS is the difference of two calls to the shared HD formula, the same
// expression NNAction::metropolis_stencil builds on the CPU from action_kernel.
template <class T>
class Phi4DsFunctor {
public:
    using element = T;
    RETICOLO_HD Phi4DsFunctor(T kappa, T lambda) : kappa_{kappa}, lambda_{lambda} {}

    RETICOLO_HD void init(T /*self*/) { nbrs_ = T{0}; }
    RETICOLO_HD void accumulate(int /*mu*/, T nbr) { nbrs_ += nbr; }
    [[nodiscard]] RETICOLO_HD double ds(T old_v, T prop) const {
        return static_cast<double>(
            action::formula::phi4_action_site<T>(prop, nbrs_, kappa_, lambda_) -
            action::formula::phi4_action_site<T>(old_v, nbrs_, kappa_, lambda_));
    }

private:
    T kappa_;
    T lambda_;
    T nbrs_ = T{0};
};

// Maps action::Phi4 to the device launchers DeviceAction calls (the primary
// template lives in device_functors.hpp). A scalar action's trait wraps the
// site-stencil skeletons (site_launchers.hpp) with its functor pair; a gauge
// action's trait wraps the plaquette skeletons instead. DeviceAction delegates
// to whichever the type resolves to and never branches.
template <class T>
struct device_functors<action::Phi4<T>> {
    static void compute_force(action::Phi4<T> const& a,
                              T const* field,
                              T* force,
                              DeviceTopology const& topo,
                              cudaStream_t s) {
        impl::site_force(Phi4ForceFunctor<T>{a.kappa, a.lambda}, field, force, topo, s);
    }
    [[nodiscard]] static double s_full(action::Phi4<T> const& a,
                                       T const* field,
                                       double* scratch,
                                       DeviceTopology const& topo,
                                       cudaStream_t s) {
        return impl::site_s_full(Phi4EnergyFunctor<T>{a.kappa, a.lambda}, field, scratch, topo, s);
    }
    static void s_full_into(double* out,
                            action::Phi4<T> const& a,
                            T const* field,
                            double* scratch,
                            double* partials,
                            DeviceTopology const& topo,
                            cudaStream_t s) {
        impl::site_s_full_into(
            out, Phi4EnergyFunctor<T>{a.kappa, a.lambda}, field, scratch, partials, topo, s);
    }
    static void sample_momenta(T* mom,
                               long n,
                               DeviceTopology const& topo,
                               std::uint64_t seed,
                               std::uint64_t const* traj,
                               cudaStream_t s) {
        impl::site_sample_momenta(mom, n, topo, seed, traj, s);
    }
    // Local-update path (cuda::Metropolis) — the same trait shape as
    // compute_force: bind the action's ds functor, hand it to the skeleton.
    static void metropolis_stencil(action::Phi4<T> const& a,
                                   T* field,
                                   DeviceTopology const& topo,
                                   int color,
                                   double sigma,
                                   std::uint64_t seed,
                                   std::uint64_t const* sweep,
                                   double* ds_out,
                                   double* acc_out,
                                   cudaStream_t s) {
        impl::site_metropolis(Phi4DsFunctor<T>{a.kappa, a.lambda},
                              field,
                              topo,
                              color,
                              sigma,
                              seed,
                              sweep,
                              ds_out,
                              acc_out,
                              s);
    }
    // A site field's elementary move is per site: the two checkerboard parities.
    [[nodiscard]] static int n_colors(DeviceTopology const& /*topo*/) { return 2; }
    // Opt-in fused path (dual-output stencil) — detected by cuda::DeviceAction and
    // used by the LLR WindowedAction to skip the redundant base-S reduction.
    static void s_full_and_force(double* out,
                                 action::Phi4<T> const& a,
                                 T const* field,
                                 T* force,
                                 double* scratch,
                                 double* partials,
                                 DeviceTopology const& topo,
                                 cudaStream_t s) {
        impl::site_s_full_and_force(out,
                                    Phi4ForceEnergyFunctor<T>{a.kappa, a.lambda},
                                    field,
                                    force,
                                    scratch,
                                    partials,
                                    topo,
                                    s);
    }
};

}  // namespace reticolo::cuda
