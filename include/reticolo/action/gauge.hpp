#pragma once

// Aggregator for the GAUGE action family — link fields on a
// `MatrixLinkLattice<G, T>`, generic over the gauge group G (Abelian U(1) or
// matrix SU(N)). `Wilson<G>` is the one plaquette leaf, deriving from
// `GaugeAction`. The gauge-group models themselves are a general primitive
// under `<reticolo/math/group/>` (used by the HMC integrator too), included
// separately.

// NOLINTBEGIN(misc-include-cleaner): re-exports are the point of the aggregator.
#include <reticolo/action/gauge/wilson.hpp>
// NOLINTEND(misc-include-cleaner)
