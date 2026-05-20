#pragma once

#include <reticolo/action/detail/concepts.hpp>
#include <reticolo/algorithm/integ_ops.hpp>
#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/link_lattice.hpp>

#include <cstddef>
#include <string_view>

namespace reticolo::alg::integ {

// =============================================================================
//  Symplectic integrators for HMC.
//
//  Every integrator is a stateless tag class with a single static template
//  `run(action, field, mom, force, tau, n_md)`. The HMC driver is templated on
//  the integrator type — there is no runtime switch in the trajectory loop,
//  so the optimiser can inline the integrator body alongside the action force.
//
//  Field-generic: works for scalar `Lattice<F>` and link `LinkLattice<F>`
//  unchanged. `flat_size(field)` overloads on the field type to give the
//  right element count; the fused-kick concept check covers both
//  `HasFusedKick` (scalar) and `HasLinkFusedKick` (gauge).
//
//  Convention: tau = total trajectory time, n_md = number of MD steps,
//  dt = tau / n_md. `force` is scratch storage owned by the HMC driver; the
//  integrator writes through it and leaves it in an unspecified state on exit.
// =============================================================================

namespace detail {

template <class A, class Field, class F = typename Field::value_type>
[[gnu::always_inline]] inline void
kick_(A const& action, Field& field, Field& mom, Field& force, double k_dt) noexcept {
    // Generic fused-kick dispatch — accepts whatever Field type the action
    // actually operates on (Lattice<F>, LinkLattice<F>, MatrixLinkLattice<G,T>).
    // The legacy scalar/link concepts are intentionally subsumed by this
    // requires-expression so adding a new field type doesn't require touching
    // every integrator concept.
    if constexpr (requires(A const& a, Field const& cf, Field& mf, F k) {
                      a.compute_force_and_kick(cf, mf, k);
                  }) {
        action.compute_force_and_kick(field, mom, static_cast<F>(k_dt));
    } else {
        action.compute_force(field, force);
        kick_add(mom, force, k_dt);
    }
}

template <class Field>
[[gnu::always_inline]] inline void drift_(Field& field, Field const& mom, double c_dt) noexcept {
    drift_field(field, mom, c_dt);
}

}  // namespace detail

// Position-Verlet leapfrog. 2nd-order, 1 force evaluation per MD step
// (amortised: n_md + 1 force evals per trajectory once the half-kicks at the
// trajectory endpoints are counted). The reference scheme — minimal cost per
// step, but acceptance falls off quickly as dt grows so trajectories at a
// given tau need many small steps.
struct Leapfrog {
    static constexpr std::string_view name = "Leapfrog";

    template <class A, class Field>
    static void run(A const& action,
                    Field& field,
                    Field& mom,
                    Field& force,
                    double const tau,
                    int const n_md) noexcept {
        double const dt      = tau / static_cast<double>(n_md);
        double const half_dt = dt * 0.5;

        detail::kick_(action, field, mom, force, half_dt);
        for (int step = 0; step < n_md; ++step) {
            detail::drift_(field, mom, dt);
            detail::kick_(action, field, mom, force, step + 1 < n_md ? dt : half_dt);
        }
    }
};

// Omelyan-Mryglod-Folk 2nd-order minimum-norm integrator (2MN).
//
//   One step (timestep dt), V = momentum kick, T = position drift:
//     V(λ·dt)  T(dt/2)  V((1-2λ)·dt)  T(dt/2)  V(λ·dt)
//
// Adjacent steps fuse boundary kicks (λ·dt + λ·dt = 2λ·dt), so the amortised
// cost is 2 force evaluations per step (2·n_md + 1 per trajectory). λ is
// chosen to minimise the leading energy-error norm in the BCH expansion
// (Omelyan, Mryglod, Folk 2002, "Optimised Forest-Ruth- and Suzuki-like
// algorithms for integration of motion in many-body systems"). At a fixed
// acceptance target the leading-error coefficient is ~10× smaller than
// Leapfrog's, so dt can be ~1.4× larger — net ~1.4× speedup at the same
// trajectory length.
struct Omelyan2 {
    static constexpr std::string_view name = "Omelyan2";

    static constexpr double k_lambda = 0.1931833275037836;

    template <class A, class Field>
    static void run(A const& action,
                    Field& field,
                    Field& mom,
                    Field& force,
                    double const tau,
                    int const n_md) noexcept {
        double const dt      = tau / static_cast<double>(n_md);
        double const dt_h    = 0.5 * dt;
        double const dt_lam  = k_lambda * dt;
        double const dt_2lam = 2.0 * dt_lam;
        double const dt_mid  = (1.0 - 2.0 * k_lambda) * dt;

        detail::kick_(action, field, mom, force, dt_lam);
        for (int step = 0; step < n_md; ++step) {
            detail::drift_(field, mom, dt_h);
            detail::kick_(action, field, mom, force, dt_mid);
            detail::drift_(field, mom, dt_h);
            detail::kick_(action, field, mom, force, step + 1 < n_md ? dt_2lam : dt_lam);
        }
    }
};

// Omelyan-Mryglod-Folk 4th-order minimum-norm integrator (4MN5FP).
//
//   One step (timestep dt), in BABABABAB form (5 kicks, 4 drifts):
//     V(ρ·dt) T(θ·dt) V(λ·dt) T(μ·dt) V((1-2ρ-2λ)·dt) T(μ·dt) V(λ·dt) T(θ·dt) V(ρ·dt)
//
//   where  μ = 1/2 - θ  (drift constraint, total drift = 1).
//
// Adjacent steps fuse boundary kicks (ρ·dt + ρ·dt = 2ρ·dt). Amortised cost:
// 4 force evaluations per step (4·n_md + 1 per trajectory) — ~4× Leapfrog's
// per-step cost. But the 4th-order error scales as dt^4 instead of dt^2, so
// dt can be much larger at the same acceptance — net ~2-3× speedup vs
// Leapfrog at typical HMC operating points, more at strict acceptance
// targets.
//
// Coefficients minimise the leading-error norm (Omelyan, Mryglod, Folk 2003,
// "Symplectic analytically integrable decomposition algorithms").
struct Omelyan4 {
    static constexpr std::string_view name = "Omelyan4";

    static constexpr double k_rho    = 0.1786178958448091;
    static constexpr double k_lambda = -0.06626458266981843;
    static constexpr double k_theta  = 0.7123418310626054;
    static constexpr double k_mu     = 0.5 - k_theta;
    static constexpr double k_mid    = 1.0 - (2.0 * k_rho) - (2.0 * k_lambda);

    template <class A, class Field>
    static void run(A const& action,
                    Field& field,
                    Field& mom,
                    Field& force,
                    double const tau,
                    int const n_md) noexcept {
        double const dt       = tau / static_cast<double>(n_md);
        double const dt_rho   = k_rho * dt;
        double const dt_2rho  = 2.0 * dt_rho;
        double const dt_lam   = k_lambda * dt;
        double const dt_theta = k_theta * dt;
        double const dt_mu    = k_mu * dt;
        double const dt_mid   = k_mid * dt;

        detail::kick_(action, field, mom, force, dt_rho);
        for (int step = 0; step < n_md; ++step) {
            detail::drift_(field, mom, dt_theta);
            detail::kick_(action, field, mom, force, dt_lam);
            detail::drift_(field, mom, dt_mu);
            detail::kick_(action, field, mom, force, dt_mid);
            detail::drift_(field, mom, dt_mu);
            detail::kick_(action, field, mom, force, dt_lam);
            detail::drift_(field, mom, dt_theta);
            detail::kick_(action, field, mom, force, step + 1 < n_md ? dt_2rho : dt_rho);
        }
    }
};

}  // namespace reticolo::alg::integ
