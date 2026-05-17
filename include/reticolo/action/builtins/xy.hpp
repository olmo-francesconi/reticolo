#pragma once

#include <reticolo/action/concepts.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numbers>

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
//  HMC-friendly: the force is -dS/dtheta. Also satisfies WolffEmbeddable via
//  the axis_type / wolff_* member block below.
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

    // -------- Wolff embedding ------------------------------------------------
    //
    // The stored angle `axis` is the direction of the reflection LINE: the
    // reflection sends theta -> 2*axis - theta, which preserves the component
    // of the spin parallel to that line and flips the perpendicular component.
    // The perpendicular unit vector therefore plays the role of Wolff's n,
    // and n·phi(theta) = sin(theta - axis), giving
    //
    //   p = 1 - exp(min(0, -2 * beta * sin(theta_x - axis) * sin(theta_y - axis)))
    //
    // i.e. the O(2) specialisation of Wolff (1989).
    using axis_type = T;

    template <class R>
        requires Rng<R>
    [[nodiscard]] axis_type wolff_random_axis(R& rng) const noexcept {
        return static_cast<T>(2.0 * std::numbers::pi * rng.uniform());
    }

    [[nodiscard]] T wolff_reflect(T theta, axis_type const& axis) const noexcept {
        return (T{2} * axis) - theta;
    }

    [[nodiscard]] double wolff_link_p(T theta_x, T theta_y, axis_type const& axis) const noexcept {
        double const sx = std::sin(static_cast<double>(theta_x - axis));
        double const sy = std::sin(static_cast<double>(theta_y - axis));
        double const w  = -2.0 * static_cast<double>(beta) * sx * sy;
        return 1.0 - std::exp(std::min(0.0, w));
    }
};

}  // namespace reticolo::action
