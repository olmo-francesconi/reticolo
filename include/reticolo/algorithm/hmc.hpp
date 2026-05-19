#pragma once

#include <reticolo/action/detail/concepts.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/link_lattice.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
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

// Field-agnostic action gate: any action with `s_full(field)` and
// `compute_force(field, force)` qualifies. The existing scalar/U(1) actions
// satisfy this via `HasSEff`+`HasForce` / `HasLinkSEff`+`HasLinkForce`; the
// new `Wilson<G>` over `MatrixLinkLattice<G, T>` satisfies it directly.
template <class A, class Field>
concept HmcAction = requires(A const& a, Field const& l, Field& f) {
    { a.s_full(l) } -> std::convertible_to<double>;
    { a.compute_force(l, f) };
};

// Compile-time tag: Field is a MatrixLinkLattice (carries a Group type alias).
template <class Field>
concept MatrixLinkField = requires { typename Field::group_type; };

template <class A,
          class R,
          class Integrator = integ::Leapfrog,
          class Field      = Lattice<typename A::value_type>,
          class F          = typename A::value_type>
    requires HmcAction<A, Field> && Rng<R>
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
        double const kin0 = kinetic_();
        double const s0   = static_cast<double>(action_.s_full(field_));
        double const h0   = kin0 + s0;

        Integrator::run(action_, field_, mom_, force_, tau_, n_md_);

        double const kin1 = kinetic_();
        double const s1   = static_cast<double>(action_.s_full(field_));
        double const h1   = kin1 + s1;
        // NOLINTNEXTLINE(readability-identifier-naming) physics convention
        double const dH = h1 - h0;

        bool const accepted = (dH <= 0.0) || (rng_.uniform() < std::exp(-dH));
        if (!accepted) {
            F* const fmut       = field_.data();
            F const* const oldp = old_field_.data();
            for (std::size_t i = 0; i < n; ++i) {
                fmut[i] = oldp[i];
            }
            // s_full now reflects the field BEFORE the trajectory — restore to s0.
            last_s_full_ = s0;
        } else {
            last_s_full_ = s1;
        }
        return {.dH = dH, .accepted = accepted};
    }

    // The s_full value of the action on the current field, as computed at the
    // end of the most recent trajectory (== s_full before the rejected step,
    // or after the accepted step). Avoids one re-sweep at the call site.
    // -∞ if no trajectory has run yet.
    [[nodiscard]] double last_s_full() const noexcept { return last_s_full_; }
    [[nodiscard]] bool has_last_s_full() const noexcept {
        return last_s_full_ == last_s_full_;  // not NaN sentinel
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
        if constexpr (MatrixLinkField<Field>) {
            // Algebra-aware momentum sampling per direction. The group writes
            // anti-hermitian elements with the right structural zeros;
            // raw normal_fill would put noise on the constrained slots.
            using Group          = typename Field::group_type;
            std::size_t const d  = mom_.ndims();
            std::size_t const ns = mom_.nsites();
            for (std::size_t mu = 0; mu < d; ++mu) {
                Group::sample_algebra_slab(mom_.mu_block_data(mu), rng_, ns);
            }
        } else if constexpr (std::is_same_v<F, double>) {
            rng_.normal_fill(mom_.data(), flat_size(mom_));
        } else if constexpr (std::is_same_v<F, std::complex<double>>) {
            std::size_t const n = flat_size(mom_);
            rng_.normal_fill(reinterpret_cast<double*>(mom_.data()), 2 * n);
        } else {
            F* const m          = mom_.data();
            std::size_t const n = flat_size(mom_);
            for (std::size_t i = 0; i < n; ++i) {
                m[i] = static_cast<F>(rng_.normal());
            }
        }
    }

    [[nodiscard]] double kinetic_() const {
        double kin = 0.0;
        if constexpr (MatrixLinkField<Field>) {
            // K = (1/2)·Tr(P†P) = ‖h‖² for SU(N) with P = i·(h·σ). Hamilton's
            // dU/dt = P·U gives ∂K/∂P = P → K = (1/2)·Tr(P†P) = ‖h‖² (no extra
            // 1/2 here). For exp(-K) detailed balance the algebra coords are
            // sampled with variance 1/2 (see SU2::sample_algebra_slab).
            using Group          = typename Field::group_type;
            std::size_t const d  = mom_.ndims();
            std::size_t const ns = mom_.nsites();
            for (std::size_t mu = 0; mu < d; ++mu) {
                kin += Group::kinetic_slab(mom_.mu_block_data(mu), ns);
            }
        } else {
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
        }
        return kin;
    }

    static constexpr bool is_complex_v_ =
        requires(F f) { std::norm(f); } && !std::is_arithmetic_v<F>;

    A const& action_;
    Field& field_;
    R& rng_;
    Field mom_;
    Field force_;
    Field old_field_;
    double tau_;
    int n_md_;
    // last s_full of `field_` after the most recent trajectory. NaN until the
    // first trajectory runs. Used by Replica / Exchange so they can read the
    // current action without re-sweeping.
    double last_s_full_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace reticolo::alg
