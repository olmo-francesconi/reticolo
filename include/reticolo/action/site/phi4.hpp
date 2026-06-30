#pragma once

#include <reticolo/action/detail/site/helpers.hpp>
#include <reticolo/action/detail/site/phi4_formula.hpp>
#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/log.hpp>

#include <cstddef>
#include <limits>
#include <vector>

namespace reticolo::action {

// Phi^4 scalar action on a hypercubic (periodic) lattice.
//
//   S = sum_x  [  -2 kappa phi(x) sum_{mu>0} phi(x+mu)
//                + phi(x)^2
//                + lambda (phi(x)^2 - 1)^2 ]
//
// Each nearest-neighbour bond appears once in `s_full` (positive-mu
// convention).
//
// MD force is `force(x) = -dS/dphi(x)`. The hot kernels (`s_full`,
// `compute_force`) hoist raw pointers and run a tight counter-indexed
// loop with no per-iteration member-access — see the body for the layout
// the compiler vectorises around.

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

    // The per-site contribution is evaluated in the field's scalar type `T`
    // (so float lattices keep their bandwidth/SIMD advantage), but the volume
    // sum accumulates in — and returns — `double`. A float sum over a large V
    // would lose ~log2(V) bits and wreck the ΔH the HMC acceptance depends on;
    // the double accumulator removes that, leaving only the per-site rounding.
    [[nodiscard]] double s_full(Lattice<T> const& l) const noexcept {
        T const k      = kappa;
        T const lam    = lambda;
        double const s = detail::reduce_fwd<T, double>(l, [k, lam](T phi, T fwd_sum) {
            return static_cast<double>(detail::phi4_action_site<T>(phi, fwd_sum, k, lam));
        });
        last_s_full_   = s;
        return s;
    }

    [[nodiscard]] double last_s_full() const noexcept { return last_s_full_; }
    void restore_last_s_full(double v) const noexcept { last_s_full_ = v; }

    // force(x) = -dS/dphi(x) = 2 kappa sum_{mu, +-} phi(x+mu) - 2 phi(x) - 4 lambda phi(x)
    // (phi(x)^2 - 1)
    void compute_force(Lattice<T> const& l, Lattice<T>& force) const noexcept {
        T const k    = kappa;
        T const lam  = lambda;
        T* const out = force.data();
        detail::visit_nn<T>(l, [k, lam, out](std::size_t i, T phi, T nbrs) {
            out[i] = detail::phi4_force_site<T>(phi, nbrs, k, lam);
        });
    }

    // Fused total action + force in ONE neighbour pass. The site action is
    // rebuilt from the full 2d neighbour sum the force already loads (each
    // bond counted twice, so the hopping weight halves to -kappa). Per-site
    // values are staged in a scratch buffer and reduced in a separate linear
    // pass — accumulating into a scalar inside the visit body would put a
    // loop-carried FP dependency in the force loop and serialise it. Returns
    // the action without updating the `last_s_full` cache — cache semantics
    // stay with `s_full`. Used by the LLR WindowedAction, whose force scale
    // needs S_base on every MD step.
    [[nodiscard]] double s_full_and_force(Lattice<T> const& l, Lattice<T>& force) const noexcept {
        T const k           = kappa;
        T const lam         = lambda;
        T* const out        = force.data();
        std::size_t const n = l.nsites();
        if (scratch.size() < n) {
            scratch.resize(n);
        }
        T* const sb = scratch.data();
        detail::visit_nn<T>(l, [k, lam, out, sb](std::size_t i, T phi, T nbrs) {
            T const phi2 = phi * phi;
            T const dev  = phi2 - T{1};
            out[i]       = detail::phi4_force_site<T>(phi, nbrs, k, lam);
            sb[i]        = (-k * phi * nbrs) + phi2 + (lam * dev * dev);
        });
        double s = 0.0;
        {
            RETICOLO_FP_REASSOCIATE
            for (std::size_t i = 0; i < n; ++i) {
                s += static_cast<double>(sb[i]);
            }
        }
        return s;
    }

    // Fused force + leapfrog kick: computes F(x) and applies mom(x) += k * F(x) in
    // one pass — the force lattice is never written or read.
    void compute_force_and_kick(Lattice<T> const& l, Lattice<T>& mom, T k_dt) const noexcept {
        T const k   = kappa;
        T const lam = lambda;
        T* const m  = mom.data();
        detail::visit_nn<T>(l, [k, lam, k_dt, m](std::size_t i, T phi, T nbrs) {
            m[i] += k_dt * detail::phi4_force_site<T>(phi, nbrs, k, lam);
        });
    }

    // Per-site action staging for `s_full_and_force`. Sized lazily to nsites.
    mutable std::vector<T> scratch{};

    // Mutable cache slot — keep public to preserve aggregate-init. Read via
    // `last_s_full()`, never assign directly from outside the action.
    mutable double last_s_full_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace reticolo::action
