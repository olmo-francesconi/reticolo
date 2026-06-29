#pragma once

#include <reticolo/action/detail/phi4_formula.hpp>
#include <reticolo/action/phi4.hpp>
#include <reticolo/cuda/actions/device_functors.hpp>
#include <reticolo/cuda/macros.hpp>

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

// Maps action::Phi4 to its device functor pair (the primary template lives in
// device_functors.hpp). A new device-ported action = a functor pair + one
// specialization in its own cuda/actions/<name>.hpp; cuda::DeviceAction stays
// generic.
template <class T>
struct device_functors<action::Phi4<T>> {
    using force  = Phi4ForceFunctor<T>;
    using energy = Phi4EnergyFunctor<T>;
    static force make_force(action::Phi4<T> const& a) { return {a.kappa, a.lambda}; }
    static energy make_energy(action::Phi4<T> const& a) { return {a.kappa, a.lambda}; }
};

}  // namespace reticolo::cuda
