#pragma once

#include <reticolo/action/concepts.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <cmath>
#include <complex>
#include <type_traits>

namespace reticolo::alg {

struct HmcSpec {
    double tau = 1.0;
    int n_md   = 20;
};

struct HmcStep {
    // NOLINTNEXTLINE(readability-identifier-naming) physics convention for ΔH = H_final - H_initial
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

        // Snapshot for rejection rollback. Flat-buffer copy — same memcpy
        // pattern as the gauge HMC twin.
        std::size_t const n     = field_.nsites();
        F* const old            = old_field_.data();
        F const* const fp_const = field_.data();
        for (std::size_t i = 0; i < n; ++i) {
            old[i] = fp_const[i];
        }
        double const h0 = hamiltonian_();

        Integrator::run(action_, field_, mom_, force_, tau_, n_md_);

        double const h1 = hamiltonian_();
        // NOLINTNEXTLINE(readability-identifier-naming) physics convention
        double const dH = h1 - h0;

        bool const accepted = (dH <= 0.0) || (rng_.uniform() < std::exp(-dH));
        if (!accepted) {
            F* const fp         = field_.data();
            F const* const oldp = old_field_.data();
            for (std::size_t i = 0; i < n; ++i) {
                fp[i] = oldp[i];
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
        // Fast paths fill the lattice storage in one call to the batched
        // normal generator. Complex F is layout-compatible with F::value_type[2]
        // (mandated by §29.5.4): independent N(0,1) draws for Re and Im give
        // the correct sampling of K = ½ Σ |p|² = ½ Σ (p_re² + p_im²).
        if constexpr (std::is_same_v<F, double>) {
            rng_.normal_fill(mom_.data(), mom_.nsites());
        } else if constexpr (std::is_same_v<F, std::complex<double>>) {
            rng_.normal_fill(reinterpret_cast<double*>(mom_.data()), 2 * mom_.nsites());
        } else {
            for (Site x : field_.sites()) {
                mom_[x] = static_cast<F>(rng_.normal());
            }
        }
    }

    [[nodiscard]] double hamiltonian_() const {
        double kin          = 0.0;
        F const* const p    = mom_.data();
        std::size_t const n = mom_.nsites();
        if constexpr (is_complex_v_) {
            for (std::size_t i = 0; i < n; ++i) {
                kin += std::norm(p[i]);
            }
        } else {
            for (std::size_t i = 0; i < n; ++i) {
                auto const pi = static_cast<double>(p[i]);
                kin += pi * pi;
            }
        }
        kin *= 0.5;
        return kin + static_cast<double>(action_.s_full(field_));
    }

    static constexpr bool is_complex_v_ =
        requires(F f) { std::norm(f); } && !std::is_arithmetic_v<F>;

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
