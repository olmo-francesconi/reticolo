#pragma once

#include <reticolo/action/detail/phi6_formula.hpp>
#include <reticolo/action/phi6.hpp>
#include <reticolo/cuda/actions/device_functors.hpp>
#include <reticolo/cuda/macros.hpp>

// Device per-site functors for Phi6 + the host-action → device-functor trait.
// Same scalar device protocol as Phi4 (init / accumulate(mu, nbr) / finalize);
// the per-site math is the shared HD formula in action::detail (phi6_formula.hpp),
// one source of truth with the CPU action::Phi6.

namespace reticolo::cuda {

template <class T>
class Phi6ForceFunctor {
public:
    using element = T;
    RETICOLO_HD Phi6ForceFunctor(T kappa, T lambda, T g6)
        : kappa_{kappa}, lambda_{lambda}, g6_{g6} {}

    RETICOLO_HD void init(T self) {
        phi_  = self;
        nbrs_ = T{0};
    }
    RETICOLO_HD void accumulate(int /*mu*/, T nbr) { nbrs_ += nbr; }
    [[nodiscard]] RETICOLO_HD T finalize() const {
        return action::detail::phi6_force_site<T>(phi_, nbrs_, kappa_, lambda_, g6_);
    }

private:
    T kappa_;
    T lambda_;
    T g6_;
    T phi_  = T{0};
    T nbrs_ = T{0};
};

template <class T>
class Phi6EnergyFunctor {
public:
    using element = T;
    RETICOLO_HD Phi6EnergyFunctor(T kappa, T lambda, T g6)
        : kappa_{kappa}, lambda_{lambda}, g6_{g6} {}

    RETICOLO_HD void init(T self) {
        phi_ = self;
        fwd_ = T{0};
    }
    RETICOLO_HD void accumulate(int /*mu*/, T nbr) { fwd_ += nbr; }
    [[nodiscard]] RETICOLO_HD double finalize() const {
        return static_cast<double>(
            action::detail::phi6_action_site<T>(phi_, fwd_, kappa_, lambda_, g6_));
    }

private:
    T kappa_;
    T lambda_;
    T g6_;
    T phi_ = T{0};
    T fwd_ = T{0};
};

template <class T>
struct device_functors<action::Phi6<T>> {
    using force  = Phi6ForceFunctor<T>;
    using energy = Phi6EnergyFunctor<T>;
    static force make_force(action::Phi6<T> const& a) { return {a.kappa, a.lambda, a.g6}; }
    static energy make_energy(action::Phi6<T> const& a) { return {a.kappa, a.lambda, a.g6}; }
};

}  // namespace reticolo::cuda
