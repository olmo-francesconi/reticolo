#pragma once

// Public umbrella include. As milestones land this will grow to cover
// actions, algorithms, observers, and IO. At M2 it exposes the core
// types: lattice + indexing + site + bc + rng + log.

// NOLINTBEGIN(misc-include-cleaner): re-exports are the whole point of the umbrella.
#include <reticolo/action/builtins/bose_gas.hpp>
#include <reticolo/action/builtins/on_sigma.hpp>
#include <reticolo/action/builtins/phi4.hpp>
#include <reticolo/action/builtins/phi6.hpp>
#include <reticolo/action/builtins/sine_gordon.hpp>
#include <reticolo/action/builtins/xy.hpp>
#include <reticolo/action/concepts.hpp>
#include <reticolo/action/helpers.hpp>
#include <reticolo/algorithm/hmc.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/algorithm/metropolis.hpp>
#include <reticolo/algorithm/wolff.hpp>
#include <reticolo/cli/parser.hpp>
#include <reticolo/core/indexing.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/link_lattice.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/gauge/algorithm/hmc.hpp>
#include <reticolo/gauge/builtins/compact_u1.hpp>
#include <reticolo/gauge/concepts.hpp>
#include <reticolo/gauge/llr/replica.hpp>
#include <reticolo/gauge/llr/windowed_action.hpp>
#include <reticolo/io/writer.hpp>
#include <reticolo/llr/exchange.hpp>
#include <reticolo/llr/replica.hpp>
#include <reticolo/llr/update_a.hpp>
#include <reticolo/llr/windowed_action.hpp>
#include <reticolo/obs/analysis.hpp>
#include <reticolo/obs/catalog.hpp>
#include <reticolo/obs/concepts.hpp>
// NOLINTEND(misc-include-cleaner)

// Short namespace aliases. One `using namespace reticolo;` per app then gives
// terse `act::Phi4`, `alg::Hmc`, `obs::mean`, `io::Writer`, `cli::Parser`.
namespace reticolo {
namespace act = action;
}  // namespace reticolo
