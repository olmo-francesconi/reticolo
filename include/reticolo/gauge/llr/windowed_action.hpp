#pragma once

#include <reticolo/core/link_lattice.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/gauge/concepts.hpp>

#include <cstddef>

namespace reticolo::gauge::llr {

// =============================================================================
//  LLR-tilted action with Gaussian-penalty window for link-field gauge
//  actions. Standard Wilson convention (weight ∝ exp(-S_W)) — identical
//  formula to the scalar real-LLR mode A:
//
//      S_LLR    = (1 + a) * S_base + (S_base - E_n)^2 / (2 * delta^2)
//      F_LLR(x) = -(1 + a + (S_base - E_n) / delta^2) * dS_base/dtheta
//               = +(1 + a + (S_base - E_n) / delta^2) * F_base(x)
//
//  where F_base = -dS_base/dtheta is what `base.compute_force` returns.
//  Boltzmann factor exp(-(1+a)*S - window) = exp(-S) * exp(-a*S - window) =
//  natural * tilt * window. At a = 0 the LLR ensemble is the natural
//  Boltzmann restricted to the window — matches the scalar twin.
//
//  Fixed-point <S - E_n> = 0 gives a* = d ln Ω/dE |_{E_n} - 1 where Ω is
//  the microcanonical density of states. Then d ln rho_natural/dE = a*, so
//  trapezoidal integration of a(E_n) reconstructs ln rho_natural directly.
//  Initialise a = 0 at the natural peak.
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
        // S_LLR = (1+a)*S + (S-E_n)^2/(2*delta^2).
        return ((T{1} + a) * s) + (ds * ds * inv);
    }

    void compute_force(field_type const& l, field_type& force) const noexcept {
        base.compute_force(l, force);
        T const s           = base.s_full(l);
        T const scale       = T{1} + a + ((s - E_n) / (delta * delta));
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
        T const scale = T{1} + a + ((s - E_n) / (delta * delta));
        base.compute_force_and_kick(l, mom, k_dt * scale);
    }
};

}  // namespace reticolo::gauge::llr
