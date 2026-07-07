#pragma once

#include <reticolo/action/site/formula/phi4_formula.hpp>
#include <reticolo/action/site/phi4.hpp>
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
