#pragma once

#include <reticolo/action/site/formula/phi4_formula.hpp>
#include <reticolo/action/site/site_action.hpp>
#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/log.hpp>

#include <cstddef>
#include <utility>

namespace reticolo::action {

// Phi^4 scalar action on a hypercubic (periodic) lattice.
//
//   S = sum_x  [  -2 kappa phi(x) sum_{mu>0} phi(x+mu)
//                + phi(x)^2
//                + lambda (phi(x)^2 - 1)^2 ]
//
// Each nearest-neighbour bond appears once in `s_full` (positive-mu
// convention). All the physics lives in the shared `detail/phi4_formula.hpp`
// functions (host+device); the loop shells, cache and LLR fast-path come from
// `SiteAction`. This struct is just the couplings + the kernel binds.

template <class T = double>
struct Phi4 : SiteAction<Phi4<T>, T> {
    using value_type = T;

    T kappa  = T{0};
    T lambda = T{0};

    void describe(log::Entry& e) const {
        e.line("Phi4<{}>", scalar_name<T>());
        e.param("κ={:.3f}", kappa);
        e.param("λ={:.3f}", lambda);
    }

    // force(x) = -dS/dphi(x) = 2 kappa sum_{mu, +-} phi(x+mu) - 2 phi(x)
    //          - 4 lambda phi(x) (phi(x)^2 - 1)
    [[nodiscard]] auto force_kernel() const noexcept {
        return [k = kappa, lam = lambda](std::size_t /*i*/, T phi, T nbrs) {
            return formula::phi4_force_site<T>(phi, nbrs, k, lam);
        };
    }

    // Per-site action from phi and the forward (positive-mu) neighbour sum.
    [[nodiscard]] auto action_kernel() const noexcept {
        return [k = kappa, lam = lambda](T phi, T fwd) {
            return formula::phi4_action_site<T>(phi, fwd, k, lam);
        };
    }

    // Fused total action + force in ONE neighbour pass (LLR fast-path). The
    // per-site action is rebuilt from the FULL 2d neighbour sum the force already
    // loads — each bond counted twice, so the hopping weight halves to -kappa —
    // matching s_full bit-for-bit. Presence of this member is what the LLR
    // WindowedAction probes for; the staging/reduction is shared in the base.
    [[nodiscard]] double s_full_and_force(Lattice<T> const& l, Lattice<T>& force) const noexcept {
        return this->staged_force_energy(
            l, force, [k = kappa, lam = lambda](std::size_t /*i*/, T phi, T nbrs) {
                T const phi2 = phi * phi;
                T const dev  = phi2 - T{1};
                T const f    = formula::phi4_force_site_p2<T>(phi, phi2, nbrs, k, lam);
                T const s    = (-k * phi * nbrs) + phi2 + (lam * dev * dev);
                return std::pair<T, T>{f, s};
            });
    }
};

}  // namespace reticolo::action
