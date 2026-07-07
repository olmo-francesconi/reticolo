#pragma once

#include <reticolo/action/detail/cache.hpp>

// GaugeAction<Derived> — the common interface for gauge (link-field) actions,
// the gauge analogue of SiteAction / BondAction / ComplexAction. It owns the
// `last_s_full` cache and the HMC-concept member surface (`s_full` writes the
// cache, `compute_force` forwards); the leaf supplies the physics.
//
// Unlike the scalar families the *traversal* cannot live in the base: a gauge
// action sweeps plaquette planes over a direction-major link field, and the
// per-plane work is fundamentally different per field type — U(1) batches four
// signed angles through a Sleef cos/sin scratch on a `LinkLattice<T>`, while
// SU(N) batches complex matrix products on a `MatrixLinkLattice<G,T>`. So the
// leaf owns the plane loop (or delegates it to a gauge-group model G), and the
// base only requires two hooks:
//
//   double s_full_uncached(Field const& U) const;      // total S (no cache write)
//   void   force_into(Field const& U, Field& force) const;  // -dS/dlink into force
//
// The fused kick (`compute_force_and_kick`) and the LLR fast-path
// (`s_full_and_force`) stay *on the leaf*, each concept-detected by its
// consumer — putting a constrained version in the base would turn those
// present/absent probes into inherited-but-unsatisfied hard errors.
//
// This split is "general for non-Wilson": a future improved/rectangle action
// derives from GaugeAction and writes its own `s_full_uncached`/`force_into`
// with a different loop functional; `Wilson<G>` (plaquette, generic over group)
// is the one plaquette leaf that ships today (`Wilson<math::group::U1>` covers the
// Abelian case directly, with n_real_components = 1).

namespace reticolo::action::detail {

template <class Derived>
struct GaugeAction : SFullCache {
    template <class Field>
    [[nodiscard]] double s_full(Field const& U) const noexcept {
        double const s = derived_().s_full_uncached(U);
        last_s_full_   = s;
        return s;
    }

    template <class Field>
    void compute_force(Field const& U, Field& force) const noexcept {
        derived_().force_into(U, force);
    }

private:
    [[nodiscard]] Derived const& derived_() const noexcept {
        return static_cast<Derived const&>(*this);
    }
};

}  // namespace reticolo::action::detail
