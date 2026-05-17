#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/site.hpp>

#include <cstddef>

namespace reticolo::alg::integ {

// =============================================================================
//  Symplectic integrators for HMC.
//
//  Every integrator is a stateless tag class with a single static template
//  `run(action, field, mom, force, tau, n_md)`. The HMC driver is templated on
//  the integrator type — there is no runtime switch in the trajectory loop,
//  so the optimiser can inline the integrator body alongside the action force.
//
//  Convention: tau = total trajectory time, n_md = number of MD steps,
//  dt = tau / n_md. `force` is scratch storage owned by the HMC driver; the
//  integrator writes through it and leaves it in an unspecified state on exit.
// =============================================================================

// Velocity-Verlet (leapfrog) — second-order symplectic.
//
//   p_{1/2}   = p_0     + (dt/2) F(q_0)
//   q_{i+1}   = q_i     + dt p_{i+1/2}
//   p_{i+3/2} = p_{i+1/2} + dt F(q_{i+1})        for i = 0 .. n_md-2
//   p_n       = p_{n-1/2} + (dt/2) F(q_n)
//
// Total cost: n_md + 1 force evaluations.
struct Leapfrog {
    template <class A, class F>
    static void run(A const& action,
                    Lattice<F>& field,
                    Lattice<F>& mom,
                    Lattice<F>& force,
                    double const tau,
                    int const n_md) noexcept {
        double const dt      = tau / static_cast<double>(n_md);
        double const half_dt = dt * 0.5;

        action.compute_force(field, force);
        for (Site x : field.sites()) {
            mom[x] += static_cast<F>(half_dt) * force[x];
        }

        for (int step = 0; step < n_md; ++step) {
            for (Site x : field.sites()) {
                field[x] += static_cast<F>(dt) * mom[x];
            }
            action.compute_force(field, force);
            double const kick = (step < n_md - 1) ? dt : half_dt;
            for (Site x : field.sites()) {
                mom[x] += static_cast<F>(kick) * force[x];
            }
        }
    }
};

}  // namespace reticolo::alg::integ
