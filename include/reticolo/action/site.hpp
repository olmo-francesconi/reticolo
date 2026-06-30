#pragma once

// Aggregator for the site (scalar / spin) action family — actions on a
// `Lattice<F>`. Each leaf is a plain struct satisfying the HMC concepts in
// `action/detail/concepts.hpp`, with its per-site formula in
// `action/detail/<name>_formula.hpp`.

// NOLINTBEGIN(misc-include-cleaner): re-exports are the point of the aggregator.
#include <reticolo/action/site/bose_gas.hpp>
#include <reticolo/action/site/phi4.hpp>
#include <reticolo/action/site/phi6.hpp>
#include <reticolo/action/site/sine_gordon.hpp>
#include <reticolo/action/site/xy.hpp>
// NOLINTEND(misc-include-cleaner)
