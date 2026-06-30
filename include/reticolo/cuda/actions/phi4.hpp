#pragma once

#include <reticolo/action/detail/site/phi4_formula.hpp>
#include <reticolo/action/site/phi4.hpp>
#include <reticolo/cuda/actions/device_functors.hpp>
#include <reticolo/cuda/actions/site_launchers.hpp>
#include <reticolo/cuda/macros.hpp>

#include <cstdint>

// Device per-site functors for Phi4 + the host-action → device-functor trait.
//
// The functors follow the scalar device protocol: init(self) /
// accumulate(int mu, T nbr) / finalize(). Their per-site math is the SHARED HD
// formula in action::detail (phi4_formula.hpp) — one source of truth with the
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
        return action::detail::phi4_force_site<T>(phi_, nbrs_, kappa_, lambda_);
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
            action::detail::phi4_action_site<T>(phi_, fwd_, kappa_, lambda_));
    }

private:
    T kappa_;
    T lambda_;
    T phi_ = T{0};
    T fwd_ = T{0};
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
        detail::site_force(Phi4ForceFunctor<T>{a.kappa, a.lambda}, field, force, topo, s);
    }
    [[nodiscard]] static double s_full(action::Phi4<T> const& a,
                                       T const* field,
                                       double* scratch,
                                       DeviceTopology const& topo,
                                       cudaStream_t s) {
        return detail::site_s_full(
            Phi4EnergyFunctor<T>{a.kappa, a.lambda}, field, scratch, topo, s);
    }
    static void s_full_into(double* out,
                            action::Phi4<T> const& a,
                            T const* field,
                            double* scratch,
                            double* partials,
                            DeviceTopology const& topo,
                            cudaStream_t s) {
        detail::site_s_full_into(
            out, Phi4EnergyFunctor<T>{a.kappa, a.lambda}, field, scratch, partials, topo, s);
    }
    static void sample_momenta(T* mom,
                               long n,
                               DeviceTopology const& topo,
                               std::uint64_t seed,
                               std::uint64_t const* traj,
                               cudaStream_t s) {
        detail::site_sample_momenta(mom, n, topo, seed, traj, s);
    }
};

}  // namespace reticolo::cuda
