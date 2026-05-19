#pragma once

#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/link_lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/gauge/concepts.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

namespace reticolo::gauge::alg {

// The integrators live in `reticolo::alg::integ` after unification — pull
// them into this namespace so existing references like `gauge::alg::integ::Omelyan2`
// and the in-class default `integ::Leapfrog` keep resolving.
namespace integ = reticolo::alg::integ;

struct HmcSpec {
    double tau = 1.0;
    int n_md   = 20;
};

struct HmcStep {
    // NOLINTNEXTLINE(readability-identifier-naming) physics convention
    double dH     = 0.0;
    bool accepted = false;
};

// =============================================================================
//  Hybrid Monte Carlo on a link-field gauge action that exposes
//  `s_full` (HasLinkSEff) and `compute_force` (HasLinkForce).
//
//  Twin of `reticolo::alg::Hmc` but on `LinkLattice<F>`. Same value-typed
//  mom/force/old_field siblings sharing the field's pooled `Indexing`, same
//  integrator-is-a-type-parameter design, same rejection-by-flat-copy.
//
//  Kinetic term: H_kin = (1/2) sum_{(x, mu)} mom_mu(x)^2 — straight sum over
//  the flat link buffer. Gaussian momenta are sampled fresh each trajectory.
//
//  Standard Wilson convention: weight ∝ exp(-S), Hamiltonian H = K + S.
//  `compute_force` returns the force F = -dS/dtheta; the integrator kick is
//  `mom += k_dt * F`. Matches the scalar HMC sign convention exactly — this
//  twin file exists only because the field type differs (LinkLattice vs
//  Lattice). The next refactor will collapse the two.
// =============================================================================

template <class A, class R, class Integrator = integ::Leapfrog, class F = typename A::value_type>
    requires gauge::HasLinkForce<A, F> && gauge::HasLinkSEff<A, F> && Rng<R>
class Hmc {
public:
    using value_type = F;
    using integrator = Integrator;

    Hmc(A const& action, LinkLattice<F>& field, R& rng, HmcSpec const& spec)
        : action_{action}, field_{field}, rng_{rng}, mom_{field.indexing()},
          force_{field.indexing()}, old_field_{field.indexing()}, tau_{spec.tau}, n_md_{spec.n_md} {
    }

    HmcStep trajectory() {
        sample_momenta_();

        // Snapshot for rejection rollback.
        F* const old        = old_field_.data();
        F const* const fp   = field_.data();
        std::size_t const n = field_.nlinks();
        for (std::size_t i = 0; i < n; ++i) {
            old[i] = fp[i];
        }
        double const h0 = hamiltonian_();

        Integrator::run(action_, field_, mom_, force_, tau_, n_md_);

        double const h1 = hamiltonian_();
        // NOLINTNEXTLINE(readability-identifier-naming) physics convention
        double const dH = h1 - h0;

        bool const accepted = (dH <= 0.0) || (rng_.uniform() < std::exp(-dH));
        if (!accepted) {
            F* const f           = field_.data();
            F const* const o     = old_field_.data();
            std::size_t const nn = field_.nlinks();
            for (std::size_t i = 0; i < nn; ++i) {
                f[i] = o[i];
            }
        }
        return {.dH = dH, .accepted = accepted};
    }

    [[nodiscard]] HmcSpec spec() const noexcept { return {.tau = tau_, .n_md = n_md_}; }
    void set_spec(HmcSpec const& s) noexcept {
        tau_  = s.tau;
        n_md_ = s.n_md;
    }

    [[nodiscard]] LinkLattice<F>& momentum() noexcept { return mom_; }
    [[nodiscard]] LinkLattice<F> const& momentum() const noexcept { return mom_; }
    [[nodiscard]] LinkLattice<F>& force_buffer() noexcept { return force_; }

    void integrate_only(double tau, int n_md) noexcept {
        Integrator::run(action_, field_, mom_, force_, tau, n_md);
    }

private:
    void sample_momenta_() {
        if constexpr (std::is_same_v<F, double>) {
            rng_.normal_fill(mom_.data(), mom_.nlinks());
        } else {
            F* const m          = mom_.data();
            std::size_t const n = mom_.nlinks();
            for (std::size_t i = 0; i < n; ++i) {
                m[i] = static_cast<F>(rng_.normal());
            }
        }
    }

    [[nodiscard]] double hamiltonian_() const {
        double kin          = 0.0;
        F const* const p    = mom_.data();
        std::size_t const n = mom_.nlinks();
        for (std::size_t i = 0; i < n; ++i) {
            auto const pi = static_cast<double>(p[i]);
            kin += pi * pi;
        }
        kin *= 0.5;
        return kin + static_cast<double>(action_.s_full(field_));
    }

    A const& action_;
    LinkLattice<F>& field_;
    R& rng_;
    LinkLattice<F> mom_;
    LinkLattice<F> force_;
    LinkLattice<F> old_field_;
    double tau_;
    int n_md_;
};

}  // namespace reticolo::gauge::alg
