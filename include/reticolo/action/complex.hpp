#pragma once

// Aggregator for the COMPLEX action family — complex scalar fields on a
// `Lattice<std::complex<T>>` with a sign problem (S = S_R + i·S_I), e.g. the
// finite-density BoseGas. Each leaf derives from `detail::ComplexAction` and
// carries only its formula (in `complex/formula/`); see `action/concepts.hpp`.

// NOLINTBEGIN(misc-include-cleaner): re-exports are the point of the aggregator.
#include <reticolo/action/complex/bose_gas.hpp>
// NOLINTEND(misc-include-cleaner)
