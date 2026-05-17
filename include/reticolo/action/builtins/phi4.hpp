#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/site.hpp>

#include <cstddef>

namespace reticolo::action {

// =============================================================================
//  Phi^4 scalar action on a hypercubic lattice.
//
//    S = sum_x  [  -2 kappa phi(x) sum_{mu>0} phi(x+mu)
//                 + phi(x)^2
//                 + lambda (phi(x)^2 - 1)^2 ]
//
//  Each nearest-neighbour bond appears once in the global sum (positive-mu
//  convention). The single-site `s_local(x)` for the Metropolis update sums
//  ALL 2d neighbours of x — that is the contribution to S that involves
//  phi(x), with each bond touching x counted once.
//
//  The MD force is force(x) = -dS/dphi(x).
//
//  Open boundaries are supported: invalid neighbour Sites contribute zero
//  (no field-with-the-outside coupling). For all-periodic shapes the validity
//  check always passes — single well-predicted branch in the hot path.
// =============================================================================

template <class T = double>
struct Phi4 {
    using value_type = T;

    T kappa  = T{0};
    T lambda = T{0};

    [[nodiscard]] T s_local(Lattice<T> const& l, Site x) const noexcept {
        T const phi = l[x];
        T nbrs      = T{0};
        for (std::size_t mu = 0; mu < l.ndims(); ++mu) {
            Site const fwd = l.next(x, mu);
            Site const bwd = l.prev(x, mu);
            if (fwd.is_valid()) {
                nbrs += l[fwd];
            }
            if (bwd.is_valid()) {
                nbrs += l[bwd];
            }
        }
        T const phi2 = phi * phi;
        T const dev  = phi2 - T{1};
        return (T{-2} * kappa * phi * nbrs) + phi2 + (lambda * dev * dev);
    }

    [[nodiscard]] T ds_local(Lattice<T> const& l, Site x, T new_v) const noexcept {
        T const phi = l[x];
        T nbrs      = T{0};
        for (std::size_t mu = 0; mu < l.ndims(); ++mu) {
            Site const fwd = l.next(x, mu);
            Site const bwd = l.prev(x, mu);
            if (fwd.is_valid()) {
                nbrs += l[fwd];
            }
            if (bwd.is_valid()) {
                nbrs += l[bwd];
            }
        }
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
        T total = T{0};
        for (Site x : l.sites()) {
            T const phi = l[x];
            T fwd_sum   = T{0};
            for (std::size_t mu = 0; mu < l.ndims(); ++mu) {
                Site const fwd = l.next(x, mu);
                if (fwd.is_valid()) {
                    fwd_sum += l[fwd];
                }
            }
            T const phi2 = phi * phi;
            T const dev  = phi2 - T{1};
            total += (T{-2} * kappa * phi * fwd_sum) + phi2 + (lambda * dev * dev);
        }
        return total;
    }

    // force(x) = -dS/dphi(x) = 2 kappa sum_{mu, +-} phi(x+mu) - 2 phi(x) - 4 lambda phi(x)
    // (phi(x)^2 - 1)
    void compute_force(Lattice<T> const& l, Lattice<T>& force) const noexcept {
        for (Site x : l.sites()) {
            T const phi = l[x];
            T nbrs      = T{0};
            for (std::size_t mu = 0; mu < l.ndims(); ++mu) {
                Site const fwd = l.next(x, mu);
                Site const bwd = l.prev(x, mu);
                if (fwd.is_valid()) {
                    nbrs += l[fwd];
                }
                if (bwd.is_valid()) {
                    nbrs += l[bwd];
                }
            }
            T const phi2 = phi * phi;
            force[x] = (T{2} * kappa * nbrs) - (T{2} * phi) - (T{4} * lambda * phi * (phi2 - T{1}));
        }
    }
};

}  // namespace reticolo::action
