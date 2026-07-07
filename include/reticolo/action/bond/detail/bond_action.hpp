#pragma once

#include <reticolo/action/detail/traverse.hpp>
#include <reticolo/core/lattice.hpp>

#include <cstddef>
#include <limits>

// BondAction<Derived, T> — the common interface for scalar actions whose energy
// is a sum over *bonds* of a function of the two endpoint values (XY / planar
// rotor, and any O(2)-style angle model). Unlike the SiteAction family the
// per-site contribution is not "self + neighbour-sum" but a transcendental of
// the endpoint difference, so the neighbour values can't be pre-summed — each
// bond is evaluated individually.
//
// The leaf supplies the physics as couplingless bond kernels plus a scale:
//
//   auto action_bond_kernel() const;   // (T self, T nbr) -> T   (S bond, pre-scale)
//   auto force_bond_kernel()  const;   // (T self, T nbr) -> T   (-dS/dself bond term)
//   T    bond_scale()         const;   // the overall prefactor (e.g. -beta)
//
// The base owns the loop shells (forward-only for the total, both-directions for
// the force), the scale, the fused kick and the `last_s_full` cache.

namespace reticolo::action::detail {

template <class Derived, class T>
struct BondAction {
    // The bond kernel IS the per-neighbour combine: unlike a site action the
    // endpoint-difference energy can't be pre-summed, so `reduce_stencil` /
    // `visit_stencil` fold `bond(self, nbr)` over the forward (total) / all
    // (force) neighbours. The shared engine supplies the tiling + threading the
    // bond family previously lacked entirely.
    [[nodiscard]] double s_full(Lattice<T> const& l) const noexcept {
        auto bond        = derived_().action_bond_kernel();
        double const raw = reduce_stencil<T, double>(
            l, bond, [](T /*self*/, T agg) { return static_cast<double>(agg); });
        double const s = static_cast<double>(derived_().bond_scale()) * raw;
        last_s_full_   = s;
        return s;
    }

    void compute_force(Lattice<T> const& l, Lattice<T>& force) const noexcept {
        auto bond    = derived_().force_bond_kernel();
        T const sc   = derived_().bond_scale();
        T* const out = force.data();
        visit_stencil<T>(
            l, bond, [sc, out](std::size_t i, T /*self*/, T agg) { out[i] = sc * agg; });
    }

    void compute_force_and_kick(Lattice<T> const& l, Lattice<T>& mom, T k_dt) const noexcept {
        auto bond  = derived_().force_bond_kernel();
        T const sc = derived_().bond_scale();
        T* const m = mom.data();
        visit_stencil<T>(l, bond, [sc, m, k_dt](std::size_t i, T /*self*/, T agg) {
            m[i] += k_dt * (sc * agg);
        });
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
