#pragma once

#include <reticolo/action/bond/detail/bond_action.hpp>
#include <reticolo/action/bond/formula/xy_formula.hpp>
#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/log.hpp>

namespace reticolo::action {

// XY (planar rotor) model on a hypercubic (periodic) lattice. theta(x) is an
// angle and the action is
//
//   S = -beta * sum_<x,y>  cos(theta(x) - theta(y))
//
// HMC-friendly: force is -dS/dtheta. The bond math lives in
// `detail/xy_formula.hpp`; the loop shells + scale come from `detail::BondAction`.

template <class T = double>
struct Xy : detail::BondAction<Xy<T>, T> {
    using value_type = T;

    T beta = T{0};

    void describe(log::Entry& e) const {
        e.line("Xy<{}>", scalar_name<T>());
        e.param("β={:.3f}", beta);
    }

    // Bond contributions (pre-scale); the -beta prefactor is applied by the base.
    [[nodiscard]] auto action_bond_kernel() const noexcept {
        return [](T self, T nbr) { return detail::xy_action_bond<T>(self, nbr); };
    }
    [[nodiscard]] auto force_bond_kernel() const noexcept {
        return [](T self, T nbr) { return detail::xy_force_bond<T>(self, nbr); };
    }
    [[nodiscard]] T bond_scale() const noexcept { return -beta; }
};

}  // namespace reticolo::action
