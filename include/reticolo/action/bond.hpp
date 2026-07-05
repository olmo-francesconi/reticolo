#pragma once

// Aggregator for the BOND action family — scalar fields on a `Lattice<T>` whose
// energy is a sum over bonds of a function of the endpoint difference (XY /
// planar rotor). Each leaf derives from `detail::BondAction` and carries only
// its formula (in `bond/formula/`); see `action/concepts.hpp` for the contract.

// NOLINTBEGIN(misc-include-cleaner): re-exports are the point of the aggregator.
#include <reticolo/action/bond/xy.hpp>
// NOLINTEND(misc-include-cleaner)
