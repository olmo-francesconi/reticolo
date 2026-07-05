#pragma once

// Aggregator for the SITE action family — scalar fields on a `Lattice<T>` whose
// energy is "self + nearest-neighbour sum" per site (Phi4 / Phi6 / SineGordon).
// Each leaf derives from `detail::SiteAction` and carries only its formula (in
// `site/formula/`); see `action/concepts.hpp` for the HMC contract.

// NOLINTBEGIN(misc-include-cleaner): re-exports are the point of the aggregator.
#include <reticolo/action/site/phi4.hpp>
#include <reticolo/action/site/phi6.hpp>
#include <reticolo/action/site/sine_gordon.hpp>
// NOLINTEND(misc-include-cleaner)
