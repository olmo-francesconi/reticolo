#pragma once

#include <reticolo/action/detail/concepts.hpp>
#include <reticolo/action/detail/site/xy_formula.hpp>
#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/site.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace reticolo::action {

// XY (planar rotor) model on a hypercubic (periodic) lattice. theta(x) is an
// angle and the action is
//
//   S = -beta * sum_<x,y>  cos(theta(x) - theta(y))
//
// HMC-friendly: force is -dS/dtheta.

template <class T = double>
struct Xy {
    using value_type = T;

    T beta = T{0};

    void describe(log::Entry& e) const {
        e.line("Xy<{}>", scalar_name<T>());
        e.param("β={:.3f}", beta);
    }

    // Per-site cos in `T`, volume sum accumulated in (and returned as) `double`
    // — see Phi4::s_full for why the total must not stay in the field type.
    [[nodiscard]] double s_full(Lattice<T> const& l) const noexcept {
        auto const& idx              = l.indexing_ref();
        T const* data                = l.data();
        Site::value_type const* next = idx.next_data();
        std::size_t const n          = idx.nsites();
        std::size_t const d          = idx.ndims();

        double total = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            T const theta          = data[i];
            std::size_t const base = i * d;
            for (std::size_t mu = 0; mu < d; ++mu) {
                total +=
                    static_cast<double>(detail::xy_action_bond<T>(theta, data[next[base + mu]]));
            }
        }
        double const s = -static_cast<double>(beta) * total;
        last_s_full_   = s;
        return s;
    }

    [[nodiscard]] double last_s_full() const noexcept { return last_s_full_; }
    void restore_last_s_full(double v) const noexcept { last_s_full_ = v; }

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
                sum += detail::xy_force_bond<T>(theta, data[next[base + mu]]);
                sum += detail::xy_force_bond<T>(theta, data[prev[base + mu]]);
            }
            out[i] = -beta * sum;
        }
    }

    mutable double last_s_full_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace reticolo::action
