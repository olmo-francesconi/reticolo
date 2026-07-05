#pragma once

// Aggregator for the GAUGE action family — link fields on a `LinkLattice<T>`
// (Abelian) or `MatrixLinkLattice<G, T>` (matrix groups). `Wilson<G>` is the
// generic plaquette action over a gauge-group model; `CompactU1` is the
// hand-tuned U(1) path. Both derive from `detail::GaugeAction`. The gauge-group
// models themselves are a general primitive under `<reticolo/math/gauge_group/>`
// (used by the HMC integrator too), included separately.

// NOLINTBEGIN(misc-include-cleaner): re-exports are the point of the aggregator.
#include <reticolo/action/gauge/wilson.hpp>
// NOLINTEND(misc-include-cleaner)
