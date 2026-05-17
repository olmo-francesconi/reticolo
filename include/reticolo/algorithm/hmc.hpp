#pragma once

#include <reticolo/action/concepts.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <cmath>

namespace reticolo::alg {

struct HmcSpec {
    double tau = 1.0;
    int n_md   = 20;
};

struct HmcStep {
    double dH     = 0.0;
    bool accepted = false;
};

// =============================================================================
//  Hybrid Monte Carlo on a scalar action that exposes both `compute_force`
//  (HasForce) and `s_full` (HasSEff).
//
//  - The integrator is a *type parameter*: there is no runtime switch in the
//    trajectory loop, and the optimiser inlines the integrator alongside the
//    action's force kernel.
//  - `mom`, `force`, and `old_field` are value-typed `Lattice<F>` siblings
//    that share the field's `Indexing` via the shape pool — three lattices,
//    one neighbour table.
//  - On rejection the field is restored from `old_field` by copy of the flat
//    buffer; no allocation per trajectory.
//
//  Kinetic term: H_kin = (1/2) sum_x mom(x)^2. Gaussian momenta are sampled
//  fresh each trajectory.
// =============================================================================
template <class A, class R, class Integrator = integ::Leapfrog, class F = typename A::value_type>
    requires action::HasForce<A, F> && action::HasSEff<A, F> && Rng<R>
class Hmc {
public:
    using value_type = F;
    using integrator = Integrator;

    Hmc(A const& action, Lattice<F>& field, R& rng, HmcSpec const& spec)
        : action_{action}, field_{field}, rng_{rng}, mom_{field.indexing()},
          force_{field.indexing()}, old_field_{field.indexing()}, tau_{spec.tau}, n_md_{spec.n_md} {
    }

    HmcStep trajectory() {
        sample_momenta_();

        // Snapshot for rejection rollback and for the initial energy.
        for (Site x : field_.sites()) {
            old_field_[x] = field_[x];
        }
        double const h0 = hamiltonian_();

        Integrator::run(action_, field_, mom_, force_, tau_, n_md_);

        double const h1 = hamiltonian_();
        double const dH = h1 - h0;

        bool const accepted = (dH <= 0.0) || (rng_.uniform() < std::exp(-dH));
        if (!accepted) {
            for (Site x : field_.sites()) {
                field_[x] = old_field_[x];
            }
        }
        return {.dH = dH, .accepted = accepted};
    }

    [[nodiscard]] HmcSpec spec() const noexcept { return {.tau = tau_, .n_md = n_md_}; }
    void set_spec(HmcSpec const& s) noexcept {
        tau_  = s.tau;
        n_md_ = s.n_md;
    }

    // Exposed so reversibility tests can run a bare integrator without
    // the Metropolis acceptance and momentum sampling.
    [[nodiscard]] Lattice<F>& momentum() noexcept { return mom_; }
    [[nodiscard]] Lattice<F> const& momentum() const noexcept { return mom_; }
    [[nodiscard]] Lattice<F>& force_buffer() noexcept { return force_; }

    void integrate_only(double tau, int n_md) noexcept {
        Integrator::run(action_, field_, mom_, force_, tau, n_md);
    }

private:
    void sample_momenta_() {
        for (Site x : field_.sites()) {
            mom_[x] = static_cast<F>(rng_.normal());
        }
    }

    [[nodiscard]] double hamiltonian_() const {
        double kin = 0.0;
        for (Site x : field_.sites()) {
            double const p = static_cast<double>(mom_[x]);
            kin += p * p;
        }
        kin *= 0.5;
        return kin + static_cast<double>(action_.s_full(field_));
    }

    A const& action_;
    Lattice<F>& field_;
    R& rng_;
    Lattice<F> mom_;
    Lattice<F> force_;
    Lattice<F> old_field_;
    double tau_;
    int n_md_;
};

}  // namespace reticolo::alg
