#pragma once

#include <reticolo/action/helpers.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/site.hpp>

#include <cmath>

namespace reticolo::action {

// =============================================================================
//  Sine-Gordon scalar action with the same NN hopping + on-site mass term as
//  Phi4, plus a periodic potential -alpha * cos(phi):
//
//    S = sum_x [ -2 kappa phi(x) sum_{mu>0} phi(x+mu)
//                + phi(x)^2
//                - alpha * cos(phi(x)) ]
//
//  At alpha = 0 this is exactly Phi4 with lambda = 0 — verified by the test
//  suite. HMC-friendly (force is the negative gradient of S).
// =============================================================================

template <class T = double>
struct SineGordon {
    using value_type = T;

    T kappa = T{0};
    T alpha = T{0};

    [[nodiscard]] T s_local(Lattice<T> const& l, Site x) const noexcept {
        T const phi  = l[x];
        T const nbrs = nn_neighbour_sum(l, x);
        return (T{-2} * kappa * phi * nbrs) + (phi * phi) - (alpha * std::cos(phi));
    }

    [[nodiscard]] T ds_local(Lattice<T> const& l, Site x, T new_v) const noexcept {
        T const phi  = l[x];
        T const nbrs = nn_neighbour_sum(l, x);
        T const hop  = T{-2} * kappa * (new_v - phi) * nbrs;
        T const mass = (new_v * new_v) - (phi * phi);
        T const pot  = -alpha * (std::cos(new_v) - std::cos(phi));
        return hop + mass + pot;
    }

    [[nodiscard]] T s_full(Lattice<T> const& l) const noexcept {
        T total = T{0};
        for (Site const x : l.sites()) {
            T const phi     = l[x];
            T const fwd_sum = fwd_neighbour_sum(l, x);
            total += (T{-2} * kappa * phi * fwd_sum) + (phi * phi) - (alpha * std::cos(phi));
        }
        return total;
    }

    // force(x) = -dS/dphi(x)
    //         = 2 kappa sum_{mu, +-} phi(x+mu) - 2 phi(x) - alpha sin(phi(x))
    void compute_force(Lattice<T> const& l, Lattice<T>& force) const noexcept {
        for (Site const x : l.sites()) {
            T const phi  = l[x];
            T const nbrs = nn_neighbour_sum(l, x);
            force[x]     = (T{2} * kappa * nbrs) - (T{2} * phi) - (alpha * std::sin(phi));
        }
    }
};

}  // namespace reticolo::action
