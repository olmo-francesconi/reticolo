#pragma once

#include <reticolo/action/concepts.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/site.hpp>

#include <cstddef>

namespace reticolo::llr {

// =============================================================================
//  LLR-tilted action with Gaussian-penalty window.
//
//  Energy variable is the base action itself: E(phi) = S_base(phi).
//
//    S_LLR     = (1 + a) * S_base + (S_base - E_n)^2 / (2 * delta^2)
//    F_LLR(x)  = (1 + a + (S_base - E_n) / delta^2) * F_base(x)
//
//  Mutable scalars a, E_n, delta are read on every `s_full` / force call —
//  the LLR driver updates them between sampling blocks.
//
//  `s_local` / `ds_local` forward to the base action. They do *not* include
//  the LLR window correction and would be wrong for a Metropolis sampler;
//  v1 only uses HMC, which only calls `s_full` and `compute_force(_and_kick)`.
// =============================================================================

template <class Base, class T = Base::value_type>
struct WindowedAction {
    using value_type = T;

    Base const& base;
    T a = T{0};
    // NOLINTNEXTLINE(readability-identifier-naming) physics convention E_n = window centre
    T E_n   = T{0};
    T delta = T{1};

    [[nodiscard]] T s_local(Lattice<T> const& l, Site x) const noexcept {
        return base.s_local(l, x);
    }

    [[nodiscard]] T ds_local(Lattice<T> const& l, Site x, T new_v) const noexcept {
        return base.ds_local(l, x, new_v);
    }

    [[nodiscard]] T s_full(Lattice<T> const& l) const noexcept {
        T const s   = base.s_full(l);
        T const ds  = s - E_n;
        T const inv = T{1} / (T{2} * delta * delta);
        return ((T{1} + a) * s) + (ds * ds * inv);
    }

    void compute_force(Lattice<T> const& l, Lattice<T>& force) const noexcept {
        base.compute_force(l, force);
        T const s           = base.s_full(l);
        T const scale       = T{1} + a + ((s - E_n) / (delta * delta));
        T* const fp         = force.data();
        std::size_t const n = force.nsites();
        for (std::size_t i = 0; i < n; ++i) {
            fp[i] *= scale;
        }
    }

    void compute_force_and_kick(Lattice<T> const& l, Lattice<T>& mom, T k_dt) const noexcept
        requires action::HasFusedKick<Base, T>
    {
        T const s     = base.s_full(l);
        T const scale = T{1} + a + ((s - E_n) / (delta * delta));
        base.compute_force_and_kick(l, mom, k_dt * scale);
    }
};

}  // namespace reticolo::llr
