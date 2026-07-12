#pragma once

#include <reticolo/action/nn/formula/phi6_formula.hpp>
#include <reticolo/action/nn/nn_action.hpp>
#include <reticolo/core/field/field_traits.hpp>
#include <reticolo/core/log/log.hpp>

#include <cstddef>

namespace reticolo::action {

// Phi^6 scalar action — same hopping + on-site phi^4 structure as Phi4 plus
// a `g6 * phi^6` term per site:
//
//   S = sum_x  [  -2 kappa phi(x) sum_{mu>0} phi(x+mu)
//               + phi(x)^2
//               + lambda * (phi(x)^2 - 1)^2
//               + g6     * phi(x)^6 ]
//
// At g6 = 0 this reduces exactly to Phi4 — verified by the physics suite. All
// physics is in `detail/phi6_formula.hpp`; the machinery is `NNAction`.

template <class T = double>
struct Phi6 : NNAction<Phi6<T>, T> {
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

    // force(x) = -dS/dphi(x)
    //         = 2 kappa sum_{mu, +-} phi(x+mu) - 2 phi(x)
    //           - 4 lambda phi(x) (phi(x)^2 - 1) - 6 g6 phi(x)^5
    [[nodiscard]] auto force_kernel() const noexcept {
        return [k = kappa, lam = lambda, g = g6](std::size_t /*i*/, T phi, T nbrs) {
            return formula::phi6_force_site<T>(phi, nbrs, k, lam, g);
        };
    }

    [[nodiscard]] auto action_kernel() const noexcept {
        return [k = kappa, lam = lambda, g = g6](T phi, T fwd) {
            return formula::phi6_action_site<T>(phi, fwd, k, lam, g);
        };
    }
};

}  // namespace reticolo::action
