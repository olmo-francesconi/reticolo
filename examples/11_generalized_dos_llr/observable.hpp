#pragma once

#include <reticolo/reticolo.hpp>

#include <cstddef>
#include <utility>

// The observable whose density of states this example reconstructs. The whole
// point of the generalised windowing is that Q is a FREE CHOICE — the base
// action you sample and the observable you constrain are independent building
// blocks. An observable is just an action leaf: define per-site formula kernels
// and its value/gradient run through the shared parallel sweep engine, no
// hand-rolled lattice loop.
//
// Here Q = the field amplitude  Σ_x φ(x)²   (per-site value φ², −dQ/dφ = −2φ).
// Swap this leaf for any other observable — magnetization Σφ, a two-point
// function, a gauge topological charge — and the same LLR machine reconstructs
// its DoS.

namespace example {

template <class T = double>
struct FieldAmplitude : reticolo::action::NNAction<FieldAmplitude<T>, T> {
    using value_type = T;
    [[nodiscard]] auto action_kernel() const noexcept {
        return [](T self, T /*agg*/) { return self * self; };  // Q_site = φ²
    }
    [[nodiscard]] auto force_kernel() const noexcept {
        return [](std::size_t /*i*/, T self, T /*agg*/) { return T{-2} * self; };  // −dQ/dφ
    }

    // Q and −dQ/dφ in ONE sweep. WindowedAction probes for this member and, when
    // present, halves the lattice passes per MD force evaluation (the window
    // needs both the constraint's value and its gradient every step). Exact for a
    // site-local observable like this one: both kernels ignore the neighbour
    // aggregate, so the fused pass evaluates the identical per-site expressions.
    [[nodiscard]] double s_full_and_force(reticolo::Lattice<T> const& l,
                                          reticolo::Lattice<T>& force) const noexcept {
        return this->staged_force_energy(l, force, [](std::size_t /*i*/, T self, T /*agg*/) {
            return std::pair<T, T>{T{-2} * self, self * self};
        });
    }
};

}  // namespace example
