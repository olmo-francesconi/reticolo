#pragma once

// Aggregator for the gauge (link-field) action family — actions on a
// `LinkLattice<F>` (Abelian) or `MatrixLinkLattice<G, F>` (matrix groups).
// `CompactU1` is the hand-tuned U(1) path; `Wilson<G>` is the generic
// Wilson action over the `gauge_group/` models. Both satisfy the same
// field-agnostic HMC concepts in `action/detail/concepts.hpp`.

// NOLINTBEGIN(misc-include-cleaner): re-exports are the point of the aggregator.
#include <reticolo/action/gauge/compact_u1.hpp>
#include <reticolo/action/gauge/wilson.hpp>
// NOLINTEND(misc-include-cleaner)
