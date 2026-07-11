#pragma once

#include <reticolo/action/nn/formula/xy_formula.hpp>
#include <reticolo/action/nn/nn_action.hpp>
#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/log.hpp>

#include <cstddef>

namespace reticolo::action {

// XY (planar rotor) model on a hypercubic (periodic) lattice. theta(x) is an
// angle and the action is
//
//   S = -beta * sum_<x,y>  cos(theta(x) - theta(y))
//
// The NN "operation" here is a per-bond transcendental of the endpoint
// difference (unlike the site actions' identity combine), so it declares
// `action_combine`/`force_combine`. There is no per-site self term, so the
// finalizers pass the aggregate through; the -beta prefactor scales the total
// action (post-reduce, via `action_scale`) and each force site. The bond math
// lives in `detail/xy_formula.hpp`.

template <class T = double>
struct Xy : NNAction<Xy<T>, T> {
    using value_type = T;

    T beta = T{0};

    void describe(log::Entry& e) const {
        e.line("Xy<{}>", scalar_name<T>());
        e.param("β={:.3f}", beta);
    }

    // Per-bond combines — the NN operation: cos / sin of the endpoint difference.
    [[nodiscard]] auto action_combine() const noexcept {
        return [](T self, T nbr) { return formula::xy_action_bond<T>(self, nbr); };
    }
    [[nodiscard]] auto force_combine() const noexcept {
        return [](T self, T nbr) { return formula::xy_force_bond<T>(self, nbr); };
    }

    // Finalizers: no self term, so the site value IS the aggregate. -beta scales
    // the total action (kept a post-reduce multiply, so s_full is bit-identical
    // to the old BondAction) and each per-site force.
    [[nodiscard]] auto action_kernel() const noexcept {
        return [](T /*self*/, T agg) { return agg; };
    }
    [[nodiscard]] auto force_kernel() const noexcept {
        return [sc = -beta](std::size_t /*i*/, T /*self*/, T agg) { return sc * agg; };
    }
    [[nodiscard]] T action_scale() const noexcept { return -beta; }
};

}  // namespace reticolo::action
