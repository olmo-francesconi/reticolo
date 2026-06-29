#pragma once

#include <reticolo/action/detail/sine_gordon_formula.hpp>
#include <reticolo/action/sine_gordon.hpp>
#include <reticolo/cuda/actions/device_functors.hpp>
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
    using force  = SineGordonForceFunctor<T>;
    using energy = SineGordonEnergyFunctor<T>;
    static force make_force(action::SineGordon<T> const& a) { return {a.kappa, a.alpha}; }
    static energy make_energy(action::SineGordon<T> const& a) { return {a.kappa, a.alpha}; }
};

}  // namespace reticolo::cuda
