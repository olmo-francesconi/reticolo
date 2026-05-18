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
//  XY (planar rotor) model on a hypercubic (periodic) lattice. theta(x) is an
//  angle and the action is
//
//    S = -beta * sum_<x,y>  cos(theta(x) - theta(y))
//
//  HMC-friendly: force is -dS/dtheta. Also satisfies WolffEmbeddable via the
//  axis_type / wolff_* member block below.
// =============================================================================

template <class T = double>
struct Xy {
    using value_type = T;

    T beta = T{0};

    [[nodiscard]] T s_local(Lattice<T> const& l, Site x) const noexcept {
        T const theta = l[x];
        T sum         = T{0};
        for (std::size_t mu = 0; mu < l.ndims(); ++mu) {
            sum += std::cos(theta - l[l.next(x, mu)]);
            sum += std::cos(theta - l[l.prev(x, mu)]);
        }
        return -beta * sum;
    }

    [[nodiscard]] T ds_local(Lattice<T> const& l, Site x, T new_v) const noexcept {
        T const theta = l[x];
        T delta_s     = T{0};
        for (std::size_t mu = 0; mu < l.ndims(); ++mu) {
            T const phi_fwd = l[l.next(x, mu)];
            T const phi_bwd = l[l.prev(x, mu)];
            delta_s += std::cos(new_v - phi_fwd) - std::cos(theta - phi_fwd);
            delta_s += std::cos(new_v - phi_bwd) - std::cos(theta - phi_bwd);
        }
        return -beta * delta_s;
    }

    [[nodiscard]] T s_full(Lattice<T> const& l) const noexcept {
        auto const& idx              = l.indexing_ref();
        T const* data                = l.data();
        Site::value_type const* next = idx.next_data();
        std::size_t const n          = idx.nsites();
        std::size_t const d          = idx.ndims();

        T total = T{0};
        for (std::size_t i = 0; i < n; ++i) {
            T const theta          = data[i];
            std::size_t const base = i * d;
            for (std::size_t mu = 0; mu < d; ++mu) {
                total += std::cos(theta - data[next[base + mu]]);
            }
        }
        return -beta * total;
    }

    // force(x) = -dS/dtheta(x) = -beta * sum_{mu, +-} sin(theta(x) - theta(x+mu)).
    void compute_force(Lattice<T> const& l, Lattice<T>& force) const noexcept {
        auto const& idx              = l.indexing_ref();
        T const* data                = l.data();
        T* out                       = force.data();
        Site::value_type const* next = idx.next_data();
        Site::value_type const* prev = idx.prev_data();
        std::size_t const n          = idx.nsites();
        std::size_t const d          = idx.ndims();

        for (std::size_t i = 0; i < n; ++i) {
            T const theta          = data[i];
            T sum                  = T{0};
            std::size_t const base = i * d;
            for (std::size_t mu = 0; mu < d; ++mu) {
                sum += std::sin(theta - data[next[base + mu]]);
                sum += std::sin(theta - data[prev[base + mu]]);
            }
            out[i] = -beta * sum;
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
        // The historical form was `1 - exp(min(0, w))`. For unfavoured bonds
        // (w >= 0) the result is 0 — skip the libm exp call entirely.
        if (w >= 0.0) {
            return 0.0;
        }
        return 1.0 - std::exp(w);
    }
};

}  // namespace reticolo::action
