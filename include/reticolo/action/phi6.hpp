#pragma once

#include <reticolo/action/detail/helpers.hpp>
#include <reticolo/action/detail/phi6_formula.hpp>
#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/log.hpp>

#include <cstddef>
#include <limits>

namespace reticolo::action {

// Phi^6 scalar action — same hopping + on-site phi^4 structure as Phi4 plus
// a `g6 * phi^6` term per site:
//
//   S = sum_x  [  -2 kappa phi(x) sum_{mu>0} phi(x+mu)
//               + phi(x)^2
//               + lambda * (phi(x)^2 - 1)^2
//               + g6     * phi(x)^6 ]
//
// At g6 = 0 this reduces exactly to Phi4 — verified by the physics suite.

template <class T = double>
struct Phi6 {
    using value_type = T;

    T kappa  = T{0};
    T lambda = T{0};
    T g6     = T{0};

    void describe(log::Entry& e) const {
        e.line("Phi6<{}>", scalar_name<T>());
        e.param("κ={:.3f}", kappa);
        e.param("λ={:.3f}", lambda);
        e.param("g₆={:.3f}", g6);
    }

    // Per-site math in `T`, volume sum accumulated in (and returned as) `double`
    // — a float sum over a large V loses ~log2(V) bits and corrupts the ΔH the
    // HMC acceptance depends on. See Phi4::s_full for the rationale.
    [[nodiscard]] double s_full(Lattice<T> const& l) const noexcept {
        T const k      = kappa;
        T const lam    = lambda;
        T const g      = g6;
        double const s = detail::reduce_fwd<T, double>(l, [k, lam, g](T phi, T fwd_sum) {
            return static_cast<double>(detail::phi6_action_site<T>(phi, fwd_sum, k, lam, g));
        });
        last_s_full_   = s;
        return s;
    }

    [[nodiscard]] double last_s_full() const noexcept { return last_s_full_; }
    void restore_last_s_full(double v) const noexcept { last_s_full_ = v; }

    // force(x) = -dS/dphi(x)
    //         = 2 kappa sum_{mu, +-} phi(x+mu) - 2 phi(x)
    //           - 4 lambda phi(x) (phi(x)^2 - 1) - 6 g6 phi(x)^5
    void compute_force(Lattice<T> const& l, Lattice<T>& force) const noexcept {
        T const k    = kappa;
        T const lam  = lambda;
        T const g    = g6;
        T* const out = force.data();
        detail::visit_nn<T>(l, [k, lam, g, out](std::size_t i, T phi, T nbrs) {
            out[i] = detail::phi6_force_site<T>(phi, nbrs, k, lam, g);
        });
    }

    // Fused force + leapfrog kick.
    void compute_force_and_kick(Lattice<T> const& l, Lattice<T>& mom, T k_dt) const noexcept {
        T const k   = kappa;
        T const lam = lambda;
        T const g   = g6;
        T* const m  = mom.data();
        detail::visit_nn<T>(l, [k, lam, g, k_dt, m](std::size_t i, T phi, T nbrs) {
            m[i] += k_dt * detail::phi6_force_site<T>(phi, nbrs, k, lam, g);
        });
    }

    mutable double last_s_full_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace reticolo::action
