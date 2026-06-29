#pragma once

#include <reticolo/action/detail/sine_gordon_formula.hpp>
#include <reticolo/action/sine_gordon.hpp>
#include <reticolo/cuda/actions/device_functors.hpp>
#include <reticolo/cuda/actions/site_launchers.hpp>
#include <reticolo/cuda/macros.hpp>

#include <cmath>

// Device per-site functors for SineGordon + the host-action → device-functor
// trait. The on-site transcendental is evaluated with std::sin/std::cos, which
// are __host__ __device__ under nvcc (a device intrinsic); the shared HD formula
// (sine_gordon_formula.hpp) does the surrounding algebra. The CPU f64 path uses
// Sleef for the same transcendental — equal to ~1 ULP (documented in the formula
// header).

namespace reticolo::cuda {

template <class T>
class SineGordonForceFunctor {
public:
    using element = T;
    RETICOLO_HD SineGordonForceFunctor(T kappa, T alpha) : kappa_{kappa}, alpha_{alpha} {}

    RETICOLO_HD void init(T self) {
        phi_  = self;
        nbrs_ = T{0};
    }
    RETICOLO_HD void accumulate(int /*mu*/, T nbr) { nbrs_ += nbr; }
    [[nodiscard]] RETICOLO_HD T finalize() const {
        return action::detail::sine_gordon_force_site<T>(
            phi_, nbrs_, std::sin(phi_), kappa_, alpha_);
    }

private:
    T kappa_;
    T alpha_;
    T phi_  = T{0};
    T nbrs_ = T{0};
};

template <class T>
class SineGordonEnergyFunctor {
public:
    using element = T;
    RETICOLO_HD SineGordonEnergyFunctor(T kappa, T alpha) : kappa_{kappa}, alpha_{alpha} {}

    RETICOLO_HD void init(T self) {
        phi_ = self;
        fwd_ = T{0};
    }
    RETICOLO_HD void accumulate(int /*mu*/, T nbr) { fwd_ += nbr; }
    [[nodiscard]] RETICOLO_HD double finalize() const {
        return static_cast<double>(
            action::detail::sine_gordon_action_site<T>(phi_, fwd_, std::cos(phi_), kappa_, alpha_));
    }

private:
    T kappa_;
    T alpha_;
    T phi_ = T{0};
    T fwd_ = T{0};
};

template <class T>
struct device_functors<action::SineGordon<T>> {
    static void compute_force(action::SineGordon<T> const& a,
                              T const* field,
                              T* force,
                              DeviceTopology const& topo,
                              cudaStream_t s) {
        detail::site_force(SineGordonForceFunctor<T>{a.kappa, a.alpha}, field, force, topo, s);
    }
    [[nodiscard]] static double s_full(action::SineGordon<T> const& a,
                                       T const* field,
                                       double* scratch,
                                       DeviceTopology const& topo,
                                       cudaStream_t s) {
        return detail::site_s_full(
            SineGordonEnergyFunctor<T>{a.kappa, a.alpha}, field, scratch, topo, s);
    }
    static void s_full_into(double* out,
                            action::SineGordon<T> const& a,
                            T const* field,
                            double* scratch,
                            double* partials,
                            DeviceTopology const& topo,
                            cudaStream_t s) {
        detail::site_s_full_into(
            out, SineGordonEnergyFunctor<T>{a.kappa, a.alpha}, field, scratch, partials, topo, s);
    }
};

}  // namespace reticolo::cuda
