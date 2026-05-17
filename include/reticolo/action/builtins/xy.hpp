#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/site.hpp>

#include <cmath>
#include <cstddef>

namespace reticolo::action {

// =============================================================================
//  XY (planar rotor) model. The field theta(x) is an angle and the action is
//
//    S = -beta * sum_<x,y>  cos(theta(x) - theta(y))
//
//  Each NN bond is counted once in s_full (positive-mu convention). The
//  per-site `s_local(x)` for Metropolis sums over all 2d neighbours of x —
//  the contribution to S that involves theta(x), with each bond touching x
//  counted once.
//
//  HMC-friendly: the force is -dS/dtheta. Wolff-cluster compatibility lands
//  alongside the Wolff updater at M9 (additional axis_type / wolff_reflect
//  members will be added there).
// =============================================================================

template <class T = double>
struct Xy {
    using value_type = T;

    T beta = T{0};

    [[nodiscard]] T s_local(Lattice<T> const& l, Site x) const noexcept {
        T const theta = l[x];
        T sum         = T{0};
        for (std::size_t mu = 0; mu < l.ndims(); ++mu) {
            Site const fwd = l.next(x, mu);
            Site const bwd = l.prev(x, mu);
            if (fwd.is_valid()) {
                sum += std::cos(theta - l[fwd]);
            }
            if (bwd.is_valid()) {
                sum += std::cos(theta - l[bwd]);
            }
        }
        return -beta * sum;
    }

    [[nodiscard]] T ds_local(Lattice<T> const& l, Site x, T new_v) const noexcept {
        T const theta = l[x];
        T delta_s     = T{0};
        for (std::size_t mu = 0; mu < l.ndims(); ++mu) {
            Site const fwd = l.next(x, mu);
            Site const bwd = l.prev(x, mu);
            if (fwd.is_valid()) {
                delta_s += std::cos(new_v - l[fwd]) - std::cos(theta - l[fwd]);
            }
            if (bwd.is_valid()) {
                delta_s += std::cos(new_v - l[bwd]) - std::cos(theta - l[bwd]);
            }
        }
        return -beta * delta_s;
    }

    [[nodiscard]] T s_full(Lattice<T> const& l) const noexcept {
        T total = T{0};
        for (Site const x : l.sites()) {
            T const theta = l[x];
            for (std::size_t mu = 0; mu < l.ndims(); ++mu) {
                Site const fwd = l.next(x, mu);
                if (fwd.is_valid()) {
                    total += std::cos(theta - l[fwd]);
                }
            }
        }
        return -beta * total;
    }

    // force(x) = -dS/dtheta(x) = -beta * sum_{mu, +-} sin(theta(x) - theta(x+mu)).
    void compute_force(Lattice<T> const& l, Lattice<T>& force) const noexcept {
        for (Site const x : l.sites()) {
            T const theta = l[x];
            T sum         = T{0};
            for (std::size_t mu = 0; mu < l.ndims(); ++mu) {
                Site const fwd = l.next(x, mu);
                Site const bwd = l.prev(x, mu);
                if (fwd.is_valid()) {
                    sum += std::sin(theta - l[fwd]);
                }
                if (bwd.is_valid()) {
                    sum += std::sin(theta - l[bwd]);
                }
            }
            force[x] = -beta * sum;
        }
    }
};

}  // namespace reticolo::action
