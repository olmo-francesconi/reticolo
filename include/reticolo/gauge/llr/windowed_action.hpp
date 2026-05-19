#pragma once

#include <reticolo/core/link_lattice.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/gauge/concepts.hpp>

#include <cstddef>

namespace reticolo::gauge::llr {

// =============================================================================
//  LLR-tilted action with Gaussian-penalty window for link-field gauge actions,
//  in the paper convention (Langfeld-Lucini-Pellegrini-Rago, arxiv:1509.08391).
//
//      Base action:        S(phi) = beta * sum_p cos(theta_p)
//      Natural Boltzmann:  exp(+S)
//      LLR target measure: exp(-a * S) * exp(-(S - E_n)^2 / (2 delta^2))
//
//  Note: at a = 0 the LLR fictitious ensemble is uniform-on-window (NOT the
//  natural HMC measure) — this is the standard paper convention.
//
//  Fixed-point condition <S - E_n> = 0 gives
//      a* = d ln g/dS |_{E_n}
//  where g(S) is the pure (Boltzmann-free) density of states. Integrating
//  the a_n's reconstructs ln g (not ln rho_natural).
//
//  To turn this into a Hamiltonian for our gauge HMC (which uses
//  weight ∝ exp(-K + s_full)) we set
//      s_full(LLR)  = -a * S(phi) - (S(phi) - E_n)^2 / (2 delta^2)
//      F(LLR)       = +ds_full/dtheta = -(a + (S - E_n)/delta^2) * F_base
// =============================================================================

template <class Base, class T = typename Base::value_type>
struct WindowedAction {
    using value_type = T;
    using field_type = typename Base::field_type;

    // Owned by value (not reference): each Replica carries its own base so any
    // mutable scratch on the action (e.g. CompactU1::scratch) is per-replica.
    // Required for OpenMP parallelism over replicas.
    Base base;
    T a = T{0};
    // NOLINTNEXTLINE(readability-identifier-naming) physics convention E_n = window centre
    T E_n   = T{0};
    T delta = T{1};

    [[nodiscard]] T s_local(field_type const& l, Site x, std::size_t mu) const noexcept {
        return base.s_local(l, x, mu);
    }

    [[nodiscard]] T ds_local(field_type const& l, Site x, std::size_t mu, T new_v) const noexcept {
        return base.ds_local(l, x, mu, new_v);
    }

    [[nodiscard]] T s_full(field_type const& l) const noexcept {
        T const s   = base.s_full(l);
        T const ds  = s - E_n;
        T const inv = T{1} / (T{2} * delta * delta);
        // s_full(LLR) = -a*S - (S-E_n)^2/(2*delta^2)
        // so that exp(-K + s_full) reproduces the LLR target above.
        return (-a * s) - (ds * ds * inv);
    }

    void compute_force(field_type const& l, field_type& force) const noexcept {
        base.compute_force(l, force);
        T const s           = base.s_full(l);
        T const scale       = -(a + ((s - E_n) / (delta * delta)));
        T* const fp         = force.data();
        std::size_t const n = force.nlinks();
        for (std::size_t i = 0; i < n; ++i) {
            fp[i] *= scale;
        }
    }

    void compute_force_and_kick(field_type const& l, field_type& mom, T k_dt) const noexcept
        requires gauge::HasLinkFusedKick<Base, T>
    {
        T const s     = base.s_full(l);
        T const scale = -(a + ((s - E_n) / (delta * delta)));
        base.compute_force_and_kick(l, mom, k_dt * scale);
    }
};

}  // namespace reticolo::gauge::llr
