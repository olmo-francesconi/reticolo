#pragma once

#include <reticolo/action/detail/xy_formula.hpp>
#include <reticolo/action/xy.hpp>
#include <reticolo/cuda/actions/device_functors.hpp>
#include <reticolo/cuda/macros.hpp>

// Device per-site functors for the XY (planar rotor) model + the host-action →
// device-functor trait. Unlike the phi-type actions, each bond contributes a
// transcendental of the angle DIFFERENCE theta(x) - theta(nbr), so the functor
// uses the per-neighbour `nbr` value (already the protocol) via the shared HD
// bond formulas (xy_formula.hpp). finalize scales the accumulated sum by -beta.

namespace reticolo::cuda {

template <class T>
class XyForceFunctor {
public:
    using element = T;
    RETICOLO_HD explicit XyForceFunctor(T beta) : beta_{beta} {}

    RETICOLO_HD void init(T self) {
        theta_ = self;
        sum_   = T{0};
    }
    RETICOLO_HD void accumulate(int /*mu*/, T nbr) {
        sum_ += action::detail::xy_force_bond<T>(theta_, nbr);
    }
    [[nodiscard]] RETICOLO_HD T finalize() const { return -beta_ * sum_; }

private:
    T beta_;
    T theta_ = T{0};
    T sum_   = T{0};
};

template <class T>
class XyEnergyFunctor {
public:
    using element = T;
    RETICOLO_HD explicit XyEnergyFunctor(T beta) : beta_{beta} {}

    RETICOLO_HD void init(T self) {
        theta_ = self;
        sum_   = T{0};
    }
    RETICOLO_HD void accumulate(int /*mu*/, T nbr) {
        sum_ += action::detail::xy_action_bond<T>(theta_, nbr);
    }
    [[nodiscard]] RETICOLO_HD double finalize() const { return static_cast<double>(-beta_ * sum_); }

private:
    T beta_;
    T theta_ = T{0};
    T sum_   = T{0};
};

template <class T>
struct device_functors<action::Xy<T>> {
    using force  = XyForceFunctor<T>;
    using energy = XyEnergyFunctor<T>;
    static force make_force(action::Xy<T> const& a) { return force{a.beta}; }
    static energy make_energy(action::Xy<T> const& a) { return energy{a.beta}; }
};

}  // namespace reticolo::cuda
