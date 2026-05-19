#pragma once

#include <reticolo/action/concepts.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/link_lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/gauge/concepts.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
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
//  Hybrid Monte Carlo — one class for both scalar (`Lattice<F>`) and link
//  (`LinkLattice<F>`) fields. The integrator is a type parameter, the field
//  type is a type parameter; the trajectory body is field-agnostic via
//  `flat_size(field)` + raw-pointer iteration.
//
//  Standard Wilson convention everywhere: weight ∝ exp(-S), H = K + S.
//  `compute_force` returns the force F = -dS/dfield (matches what the
//  integrator's `mom += k_dt * F` expects).
//
//  Kinetic term:
//    - real F:    H_kin = (1/2) sum_i mom_i^2
//    - complex F: H_kin = (1/2) sum_i |mom_i|^2  (= (Re^2 + Im^2)/2 per slot)
//
//  Momentum sampling:
//    - double:                 one batched normal_fill
//    - std::complex<double>:   reinterpret-cast to double[2] and batch-fill;
//                              gives independent N(0,1) on Re and Im, which
//                              is the correct sampling for the complex
//                              Gaussian (§29.5.4 guarantees the layout).
//    - anything else:          per-element static_cast<F>(rng_.normal())
// =============================================================================

template <class A,
          class R,
          class Integrator = integ::Leapfrog,
          class Field      = Lattice<typename A::value_type>,
          class F          = typename A::value_type>
    requires(action::HasForce<A, F> || gauge::HasLinkForce<A, F>) &&
            (action::HasSEff<A, F> || gauge::HasLinkSEff<A, F>) && Rng<R>
class Hmc {
public:
    using value_type = F;
    using integrator = Integrator;

    Hmc(A const& action, Field& field, R& rng, HmcSpec const& spec)
        : action_{action}, field_{field}, rng_{rng}, mom_{field.indexing()},
          force_{field.indexing()}, old_field_{field.indexing()}, tau_{spec.tau}, n_md_{spec.n_md} {
    }

    HmcStep trajectory() {
        sample_momenta_();

        // Snapshot for rejection rollback — flat-buffer copy.
        std::size_t const n = flat_size(field_);
        F* const old        = old_field_.data();
        F const* const fp   = field_.data();
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
            F* const fmut       = field_.data();
            F const* const oldp = old_field_.data();
            for (std::size_t i = 0; i < n; ++i) {
                fmut[i] = oldp[i];
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
    [[nodiscard]] Field& momentum() noexcept { return mom_; }
    [[nodiscard]] Field const& momentum() const noexcept { return mom_; }
    [[nodiscard]] Field& force_buffer() noexcept { return force_; }

    void integrate_only(double tau, int n_md) noexcept {
        Integrator::run(action_, field_, mom_, force_, tau, n_md);
    }

private:
    void sample_momenta_() {
        std::size_t const n = flat_size(mom_);
        if constexpr (std::is_same_v<F, double>) {
            rng_.normal_fill(mom_.data(), n);
        } else if constexpr (std::is_same_v<F, std::complex<double>>) {
            rng_.normal_fill(reinterpret_cast<double*>(mom_.data()), 2 * n);
        } else {
            F* const m = mom_.data();
            for (std::size_t i = 0; i < n; ++i) {
                m[i] = static_cast<F>(rng_.normal());
            }
        }
    }

    [[nodiscard]] double hamiltonian_() const {
        double kin          = 0.0;
        F const* const p    = mom_.data();
        std::size_t const n = flat_size(mom_);
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
    Field&   field_;
    R&       rng_;
    Field    mom_;
    Field    force_;
    Field    old_field_;
    double   tau_;
    int      n_md_;
};

}  // namespace reticolo::alg
