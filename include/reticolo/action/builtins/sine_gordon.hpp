#pragma once

#include <reticolo/action/helpers.hpp>
#include <reticolo/action/hot_loop.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/site.hpp>

#include <cmath>
#include <cstddef>

namespace reticolo::action {

// =============================================================================
//  Sine-Gordon scalar action: same NN hopping + on-site mass term as Phi4,
//  plus a periodic potential -alpha * cos(phi):
//
//    S = sum_x [ -2 kappa phi(x) sum_{mu>0} phi(x+mu)
//                + phi(x)^2
//                - alpha * cos(phi(x)) ]
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
        return ds_local_from_nbrs(l[x], new_v, nn_neighbour_sum(l, x));
    }

    [[nodiscard]] T ds_local_from_nbrs(T phi, T new_v, T nbrs) const noexcept {
        T const hop  = T{-2} * kappa * (new_v - phi) * nbrs;
        T const mass = (new_v * new_v) - (phi * phi);
        T const pot  = -alpha * (std::cos(new_v) - std::cos(phi));
        return hop + mass + pot;
    }

    [[nodiscard]] T s_full(Lattice<T> const& l) const noexcept {
        T const k   = kappa;
        T const alp = alpha;
        return detail::reduce_fwd<T>(l, [k, alp](T phi, T fwd_sum) {
            return (T{-2} * k * phi * fwd_sum) + (phi * phi) - (alp * std::cos(phi));
        });
    }

    // force(x) = -dS/dphi(x)
    //         = 2 kappa sum_{mu, +-} phi(x+mu) - 2 phi(x) - alpha sin(phi(x))
    void compute_force(Lattice<T> const& l, Lattice<T>& force) const noexcept {
        T const  k    = kappa;
        T const  alp  = alpha;
        T* const out  = force.data();
        detail::visit_nn<T>(l, [k, alp, out](std::size_t i, T phi, T nbrs) {
            out[i] = (T{2} * k * nbrs) - (T{2} * phi) - (alp * std::sin(phi));
        });
    }

    // Fused force + leapfrog kick.
    void compute_force_and_kick(Lattice<T> const& l, Lattice<T>& mom, T k_dt) const noexcept {
        T const  k    = kappa;
        T const  alp  = alpha;
        T* const m    = mom.data();
        detail::visit_nn<T>(l, [k, alp, k_dt, m](std::size_t i, T phi, T nbrs) {
            T const F = (T{2} * k * nbrs) - (T{2} * phi) - (alp * std::sin(phi));
            m[i] += k_dt * F;
        });
    }
};

}  // namespace reticolo::action
