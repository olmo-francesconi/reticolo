#pragma once

#include <reticolo/action/helpers.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/site.hpp>

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
//  At g6 = 0 this reduces exactly to Phi4 — verified as part of the M8
//  physics suite.
// =============================================================================

template <class T = double>
struct Phi6 {
    using value_type = T;

    T kappa  = T{0};
    T lambda = T{0};
    T g6     = T{0};

    [[nodiscard]] T s_local(Lattice<T> const& l, Site x) const noexcept {
        T const phi  = l[x];
        T const nbrs = nn_neighbour_sum(l, x);
        T const phi2 = phi * phi;
        T const dev  = phi2 - T{1};
        T const phi6 = phi2 * phi2 * phi2;
        return (T{-2} * kappa * phi * nbrs) + phi2 + (lambda * dev * dev) + (g6 * phi6);
    }

    [[nodiscard]] T ds_local(Lattice<T> const& l, Site x, T new_v) const noexcept {
        T const phi      = l[x];
        T const nbrs     = nn_neighbour_sum(l, x);
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
        T total            = T{0};
        auto const on_site = [this](T phi, T fwd_sum) {
            T const phi2 = phi * phi;
            T const dev  = phi2 - T{1};
            T const phi6 = phi2 * phi2 * phi2;
            return (T{-2} * kappa * phi * fwd_sum) + phi2 + (lambda * dev * dev) + (g6 * phi6);
        };
        for (Site const x : l.bulk_sites()) {
            total += on_site(l[x], fwd_neighbour_sum_unchecked(l, x));
        }
        for (Site const x : l.skin_sites()) {
            total += on_site(l[x], fwd_neighbour_sum(l, x));
        }
        return total;
    }

    // force(x) = -dS/dphi(x)
    //         = 2 kappa sum_{mu, +-} phi(x+mu) - 2 phi(x)
    //           - 4 lambda phi(x) (phi(x)^2 - 1) - 6 g6 phi(x)^5
    void compute_force(Lattice<T> const& l, Lattice<T>& force) const noexcept {
        auto const at = [&](Site x, T nbrs) {
            T const phi  = l[x];
            T const phi2 = phi * phi;
            T const phi5 = phi2 * phi2 * phi;
            force[x]     = (T{2} * kappa * nbrs) - (T{2} * phi) -
                       (T{4} * lambda * phi * (phi2 - T{1})) - (T{6} * g6 * phi5);
        };
        for (Site const x : l.bulk_sites()) {
            at(x, nn_neighbour_sum_unchecked(l, x));
        }
        for (Site const x : l.skin_sites()) {
            at(x, nn_neighbour_sum(l, x));
        }
    }
};

}  // namespace reticolo::action
