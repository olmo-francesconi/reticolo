#pragma once

#include <limits>

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
// and `CompactU1` (plaquette, hand-tuned U(1) on the lighter LinkLattice) are
// simply the two plaquette leaves that ship today.

namespace reticolo::action::detail {

template <class Derived>
struct GaugeAction {
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

    [[nodiscard]] double last_s_full() const noexcept { return last_s_full_; }
    void restore_last_s_full(double v) const noexcept { last_s_full_ = v; }

    mutable double last_s_full_ = std::numeric_limits<double>::quiet_NaN();

private:
    [[nodiscard]] Derived const& derived_() const noexcept {
        return static_cast<Derived const&>(*this);
    }
};

}  // namespace reticolo::action::detail
