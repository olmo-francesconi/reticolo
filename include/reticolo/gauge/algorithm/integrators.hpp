#pragma once

#include <reticolo/core/link_lattice.hpp>
#include <reticolo/gauge/concepts.hpp>

#include <cstddef>

namespace reticolo::gauge::alg::integ {

// =============================================================================
//  Symplectic integrators for link HMC.
//
//  Twin of `reticolo::alg::integ` but for `LinkLattice<F>` fields. Three
//  integrators: Leapfrog (2nd-order, 1 force eval/step), Omelyan2 (2MN,
//  2 evals/step), Omelyan4 (4MN5FP, 4 evals/step). Each is stateless and
//  has a single static template `run(action, field, mom, force, tau, n_md)`.
//
//  drift_ and kick_ operate on the flat link buffer — same stride-1
//  vectorisable pattern as the scalar version, but the field has
//  ndim * nsites elements instead of nsites.
// =============================================================================

namespace detail {

template <class A, class F>
[[gnu::always_inline]] inline void kick_(A const& action,
                                         LinkLattice<F>& field,
                                         LinkLattice<F>& mom,
                                         LinkLattice<F>& force,
                                         double k_dt) noexcept {
    if constexpr (gauge::HasLinkFusedKick<A, F>) {
        action.compute_force_and_kick(field, mom, static_cast<F>(k_dt));
    } else {
        action.compute_force(field, force);
        F* const m          = mom.data();
        F const* const fp   = force.data();
        std::size_t const n = mom.nlinks();
        F const kdt         = static_cast<F>(k_dt);
        for (std::size_t i = 0; i < n; ++i) {
            m[i] += kdt * fp[i];
        }
    }
}

template <class F>
[[gnu::always_inline]] inline void
drift_(LinkLattice<F>& field, LinkLattice<F> const& mom, double c_dt) noexcept {
    F* const f          = field.data();
    F const* const p    = mom.data();
    std::size_t const n = field.nlinks();
    F const cdt         = static_cast<F>(c_dt);
    for (std::size_t i = 0; i < n; ++i) {
        f[i] += cdt * p[i];
    }
}

}  // namespace detail

// 2nd-order Stoermer-Verlet (leapfrog).
struct Leapfrog {
    template <class A, class F>
    static void run(A const& action,
                    LinkLattice<F>& field,
                    LinkLattice<F>& mom,
                    LinkLattice<F>& force,
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

// Omelyan-Mryglod-Folk 2nd-order minimum-norm (2MN), 2 force evals/step.
struct Omelyan2 {
    static constexpr double k_lambda = 0.1931833275037836;

    template <class A, class F>
    static void run(A const& action,
                    LinkLattice<F>& field,
                    LinkLattice<F>& mom,
                    LinkLattice<F>& force,
                    double const tau,
                    int const n_md) noexcept {
        double const dt      = tau / static_cast<double>(n_md);
        double const dt_h    = 0.5 * dt;
        double const dt_lam  = k_lambda * dt;
        double const dt_2lam = 2.0 * dt_lam;
        double const dt_mid  = (1.0 - (2.0 * k_lambda)) * dt;

        detail::kick_(action, field, mom, force, dt_lam);
        for (int step = 0; step < n_md; ++step) {
            detail::drift_(field, mom, dt_h);
            detail::kick_(action, field, mom, force, dt_mid);
            detail::drift_(field, mom, dt_h);
            detail::kick_(action, field, mom, force, step + 1 < n_md ? dt_2lam : dt_lam);
        }
    }
};

// Omelyan-Mryglod-Folk 4th-order minimum-norm (4MN5FP), 4 force evals/step.
struct Omelyan4 {
    static constexpr double k_rho    = 0.1786178958448091;
    static constexpr double k_lambda = -0.06626458266981843;
    static constexpr double k_theta  = 0.7123418310626054;
    static constexpr double k_mu     = 0.5 - k_theta;
    static constexpr double k_mid    = 1.0 - (2.0 * k_rho) - (2.0 * k_lambda);

    template <class A, class F>
    static void run(A const& action,
                    LinkLattice<F>& field,
                    LinkLattice<F>& mom,
                    LinkLattice<F>& force,
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

}  // namespace reticolo::gauge::alg::integ
