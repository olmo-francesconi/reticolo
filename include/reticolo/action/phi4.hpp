#pragma once

#include <reticolo/action/detail/helpers.hpp>
#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/site.hpp>

#include <cstddef>
#include <limits>

namespace reticolo::action {

// =============================================================================
//  Phi^4 scalar action on a hypercubic (periodic) lattice.
//
//    S = sum_x  [  -2 kappa phi(x) sum_{mu>0} phi(x+mu)
//                 + phi(x)^2
//                 + lambda (phi(x)^2 - 1)^2 ]
//
//  Each nearest-neighbour bond appears once in `s_full` (positive-mu
//  convention). `s_local(x)` for Metropolis sums all 2d neighbours of x —
//  the contribution to S that involves phi(x), each touching bond once.
//
//  MD force is `force(x) = -dS/dphi(x)`. The hot kernels (`s_full`,
//  `compute_force`) hoist raw pointers and run a tight counter-indexed
//  loop with no per-iteration member-access — see the body for the layout
//  the compiler vectorises around.
// =============================================================================

template <class T = double>
struct Phi4 {
    using value_type = T;

    T kappa  = T{0};
    T lambda = T{0};

    void describe(log::Entry& e) const {
        e.line("Phi4<{}>", scalar_name<T>());
        e.param("κ={:.3f}", kappa);
        e.param("λ={:.3f}", lambda);
    }

    [[nodiscard]] T s_local(Lattice<T> const& l, Site x) const noexcept {
        T const phi  = l[x];
        T const nbrs = nn_neighbour_sum(l, x);
        T const phi2 = phi * phi;
        T const dev  = phi2 - T{1};
        return (T{-2} * kappa * phi * nbrs) + phi2 + (lambda * dev * dev);
    }

    [[nodiscard]] T ds_local(Lattice<T> const& l, Site x, T new_v) const noexcept {
        return ds_local_from_nbrs(l[x], new_v, nn_neighbour_sum(l, x));
    }

    // Pure-math form: ds given phi, new_v, and the unweighted sum of all 2*ndims
    // nearest-neighbour values. Used by the visit_nn Metropolis fast path.
    [[nodiscard]] T ds_local_from_nbrs(T phi, T new_v, T nbrs) const noexcept {
        T const phi2_old = phi * phi;
        T const phi2_new = new_v * new_v;
        T const dev_old  = phi2_old - T{1};
        T const dev_new  = phi2_new - T{1};
        T const hop      = T{-2} * kappa * (new_v - phi) * nbrs;
        T const onsite =
            (phi2_new - phi2_old) + (lambda * ((dev_new * dev_new) - (dev_old * dev_old)));
        return hop + onsite;
    }

    [[nodiscard]] T s_full(Lattice<T> const& l) const noexcept {
        T const k    = kappa;
        T const lam  = lambda;
        T const s    = detail::reduce_fwd<T>(l, [k, lam](T phi, T fwd_sum) {
            T const phi2 = phi * phi;
            T const dev  = phi2 - T{1};
            return (T{-2} * k * phi * fwd_sum) + phi2 + (lam * dev * dev);
        });
        last_s_full_ = s;
        return s;
    }

    [[nodiscard]] T last_s_full() const noexcept { return last_s_full_; }
    void restore_last_s_full(T v) const noexcept { last_s_full_ = v; }

    // force(x) = -dS/dphi(x) = 2 kappa sum_{mu, +-} phi(x+mu) - 2 phi(x) - 4 lambda phi(x)
    // (phi(x)^2 - 1)
    void compute_force(Lattice<T> const& l, Lattice<T>& force) const noexcept {
        T const k    = kappa;
        T const lam  = lambda;
        T* const out = force.data();
        detail::visit_nn<T>(l, [k, lam, out](std::size_t i, T phi, T nbrs) {
            T const phi2 = phi * phi;
            out[i]       = (T{2} * k * nbrs) - (T{2} * phi) - (T{4} * lam * phi * (phi2 - T{1}));
        });
    }

    // Fused force + leapfrog kick: computes F(x) and applies mom(x) += k * F(x) in
    // one pass — the force lattice is never written or read.
    void compute_force_and_kick(Lattice<T> const& l, Lattice<T>& mom, T k_dt) const noexcept {
        T const k   = kappa;
        T const lam = lambda;
        T* const m  = mom.data();
        detail::visit_nn<T>(l, [k, lam, k_dt, m](std::size_t i, T phi, T nbrs) {
            T const phi2 = phi * phi;
            T const F    = (T{2} * k * nbrs) - (T{2} * phi) - (T{4} * lam * phi * (phi2 - T{1}));
            m[i] += k_dt * F;
        });
    }

    // Mutable cache slot — keep public to preserve aggregate-init. Read via
    // `last_s_full()`, never assign directly from outside the action.
    mutable T last_s_full_ = std::numeric_limits<T>::quiet_NaN();
};

}  // namespace reticolo::action
