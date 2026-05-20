#pragma once

#include <reticolo/action/detail/helpers.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/site.hpp>

#include <cstddef>
#include <limits>

namespace reticolo::action {

// =============================================================================
//  Phi^6 scalar action — same hopping + on-site phi^4 structure as Phi4 plus
//  a `g6 * phi^6` term per site:
//
//    S = sum_x  [  -2 kappa phi(x) sum_{mu>0} phi(x+mu)
//                + phi(x)^2
//                + lambda * (phi(x)^2 - 1)^2
//                + g6     * phi(x)^6 ]
//
//  At g6 = 0 this reduces exactly to Phi4 — verified by the M8 physics suite.
// =============================================================================

template <class T = double>
struct Phi6 {
    using value_type = T;

    T kappa  = T{0};
    T lambda = T{0};
    T g6     = T{0};

    [[nodiscard]] T s_local(Lattice<T> const& l, Site x) const noexcept {
        return ds_baseline_(l[x], nn_neighbour_sum(l, x));
    }

    [[nodiscard]] T ds_local(Lattice<T> const& l, Site x, T new_v) const noexcept {
        return ds_local_from_nbrs(l[x], new_v, nn_neighbour_sum(l, x));
    }

    [[nodiscard]] T ds_local_from_nbrs(T phi, T new_v, T nbrs) const noexcept {
        T const phi2_old = phi * phi;
        T const phi2_new = new_v * new_v;
        T const dev_old  = phi2_old - T{1};
        T const dev_new  = phi2_new - T{1};
        T const phi6_old = phi2_old * phi2_old * phi2_old;
        T const phi6_new = phi2_new * phi2_new * phi2_new;
        T const hop      = T{-2} * kappa * (new_v - phi) * nbrs;
        T const onsite   = (phi2_new - phi2_old) +
                         (lambda * ((dev_new * dev_new) - (dev_old * dev_old))) +
                         (g6 * (phi6_new - phi6_old));
        return hop + onsite;
    }

    [[nodiscard]] T s_full(Lattice<T> const& l) const noexcept {
        T const k    = kappa;
        T const lam  = lambda;
        T const g    = g6;
        T const s    = detail::reduce_fwd<T>(l, [k, lam, g](T phi, T fwd_sum) {
            T const phi2 = phi * phi;
            T const dev  = phi2 - T{1};
            T const phi6 = phi2 * phi2 * phi2;
            return (T{-2} * k * phi * fwd_sum) + phi2 + (lam * dev * dev) + (g * phi6);
        });
        last_s_full_ = s;
        return s;
    }

    [[nodiscard]] T last_s_full() const noexcept { return last_s_full_; }
    void restore_last_s_full(T v) const noexcept { last_s_full_ = v; }

    // force(x) = -dS/dphi(x)
    //         = 2 kappa sum_{mu, +-} phi(x+mu) - 2 phi(x)
    //           - 4 lambda phi(x) (phi(x)^2 - 1) - 6 g6 phi(x)^5
    void compute_force(Lattice<T> const& l, Lattice<T>& force) const noexcept {
        T const k    = kappa;
        T const lam  = lambda;
        T const g    = g6;
        T* const out = force.data();
        detail::visit_nn<T>(l, [k, lam, g, out](std::size_t i, T phi, T nbrs) {
            T const phi2 = phi * phi;
            T const phi5 = phi2 * phi2 * phi;
            out[i]       = (T{2} * k * nbrs) - (T{2} * phi) - (T{4} * lam * phi * (phi2 - T{1})) -
                     (T{6} * g * phi5);
        });
    }

    // Fused force + leapfrog kick.
    void compute_force_and_kick(Lattice<T> const& l, Lattice<T>& mom, T k_dt) const noexcept {
        T const k   = kappa;
        T const lam = lambda;
        T const g   = g6;
        T* const m  = mom.data();
        detail::visit_nn<T>(l, [k, lam, g, k_dt, m](std::size_t i, T phi, T nbrs) {
            T const phi2 = phi * phi;
            T const phi5 = phi2 * phi2 * phi;
            T const F    = (T{2} * k * nbrs) - (T{2} * phi) - (T{4} * lam * phi * (phi2 - T{1})) -
                        (T{6} * g * phi5);
            m[i] += k_dt * F;
        });
    }

    mutable T last_s_full_ = std::numeric_limits<T>::quiet_NaN();

private:
    [[nodiscard]] T ds_baseline_(T phi, T nbrs) const noexcept {
        T const phi2 = phi * phi;
        T const dev  = phi2 - T{1};
        T const phi6 = phi2 * phi2 * phi2;
        return (T{-2} * kappa * phi * nbrs) + phi2 + (lambda * dev * dev) + (g6 * phi6);
    }
};

}  // namespace reticolo::action
