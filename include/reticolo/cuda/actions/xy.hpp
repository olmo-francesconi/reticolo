#pragma once

#include <reticolo/action/detail/xy_formula.hpp>
#include <reticolo/action/xy.hpp>
#include <reticolo/cuda/actions/device_functors.hpp>
#include <reticolo/cuda/actions/site_launchers.hpp>
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
    static void compute_force(action::Xy<T> const& a,
                              T const* field,
                              T* force,
                              DeviceTopology const& topo,
                              cudaStream_t s) {
        detail::site_force(XyForceFunctor<T>{a.beta}, field, force, topo, s);
    }
    [[nodiscard]] static double s_full(action::Xy<T> const& a,
                                       T const* field,
                                       double* scratch,
                                       DeviceTopology const& topo,
                                       cudaStream_t s) {
        return detail::site_s_full(XyEnergyFunctor<T>{a.beta}, field, scratch, topo, s);
    }
    static void s_full_into(double* out,
                            action::Xy<T> const& a,
                            T const* field,
                            double* scratch,
                            double* partials,
                            DeviceTopology const& topo,
                            cudaStream_t s) {
        detail::site_s_full_into(
            out, XyEnergyFunctor<T>{a.beta}, field, scratch, partials, topo, s);
    }
};

}  // namespace reticolo::cuda
