#pragma once

// Public umbrella include — covers actions, algorithms, observers, IO, and
// core types (lattice, indexing, site, bc, rng, log).

// NOLINTBEGIN(misc-include-cleaner): re-exports are the whole point of the umbrella.
#include <reticolo/action/bose_gas.hpp>
#include <reticolo/action/compact_u1.hpp>
#include <reticolo/action/detail/concepts.hpp>
#include <reticolo/action/detail/gauge_group/base.hpp>
#include <reticolo/action/detail/gauge_group/su2.hpp>
#include <reticolo/action/detail/gauge_group/su3.hpp>
#include <reticolo/action/detail/gauge_group/u1.hpp>
#include <reticolo/action/detail/helpers.hpp>
#include <reticolo/action/on_sigma.hpp>
#include <reticolo/action/phi4.hpp>
#include <reticolo/action/phi6.hpp>
#include <reticolo/action/sine_gordon.hpp>
#include <reticolo/action/wilson.hpp>
#include <reticolo/action/xy.hpp>
#include <reticolo/algorithm/hmc.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/algorithm/metropolis.hpp>
#include <reticolo/algorithm/wolff.hpp>
#include <reticolo/app/setup.hpp>
#include <reticolo/cli/parser.hpp>
#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/indexing.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/link_lattice.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/log_helpers.hpp>
#include <reticolo/core/mt19937_rng.hpp>
#include <reticolo/core/philox.hpp>
#include <reticolo/core/philox_rng.hpp>
#include <reticolo/core/ranlux_rng.hpp>
#include <reticolo/core/rng.hpp>
#include <reticolo/core/site.hpp>
#include <reticolo/io/checkpoint.hpp>
#include <reticolo/io/reader.hpp>
#include <reticolo/io/writer.hpp>
#include <reticolo/llr/driver.hpp>
#include <reticolo/llr/exchange.hpp>
#include <reticolo/llr/log.hpp>
#include <reticolo/llr/replica.hpp>
#include <reticolo/llr/smoothed_driver.hpp>
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
