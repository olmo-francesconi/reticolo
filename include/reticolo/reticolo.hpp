#pragma once

// Public umbrella include — covers actions, algorithms, observers, IO, and
// core types (lattice, indexing, site, bc, rng, log).

// NOLINTBEGIN(misc-include-cleaner): re-exports are the whole point of the umbrella.
#include <reticolo/action/complex.hpp>
#include <reticolo/action/concepts.hpp>
#include <reticolo/action/gauge.hpp>
#include <reticolo/action/nn.hpp>
#include <reticolo/action/sweep/site.hpp>
#include <reticolo/action/windowed_action.hpp>
#include <reticolo/app/setup.hpp>
#include <reticolo/cli/parser.hpp>
#include <reticolo/core/field/field_traits.hpp>
#include <reticolo/core/field/indexing.hpp>
#include <reticolo/core/field/lattice.hpp>
#include <reticolo/core/log/log.hpp>
#include <reticolo/core/log/log_helpers.hpp>
#include <reticolo/core/rng/fast_rng.hpp>
#include <reticolo/core/rng/mt19937_rng.hpp>
#include <reticolo/core/rng/philox.hpp>
#include <reticolo/core/rng/philox_rng.hpp>
#include <reticolo/core/rng/ranlxd_rng.hpp>
#include <reticolo/core/rng/rng.hpp>
#include <reticolo/core/rng/stream_set.hpp>
#include <reticolo/core/field/site.hpp>
#include <reticolo/io/checkpoint.hpp>
#include <reticolo/io/reader.hpp>
#include <reticolo/io/writer.hpp>
#include <reticolo/math/group/su2.hpp>
#include <reticolo/math/group/su3.hpp>
#include <reticolo/math/group/u1.hpp>
#include <reticolo/obs/analysis.hpp>
#include <reticolo/obs/catalog.hpp>
#include <reticolo/obs/concepts.hpp>
#include <reticolo/obs/reduce.hpp>
#include <reticolo/orch/checkpoint.hpp>
#include <reticolo/orch/concepts.hpp>
#include <reticolo/orch/ensemble.hpp>
#include <reticolo/orch/llr/exchange.hpp>
#include <reticolo/orch/llr/log.hpp>
#include <reticolo/orch/llr/orchestrator.hpp>
#include <reticolo/orch/llr/replica.hpp>
#include <reticolo/orch/llr/update_a.hpp>
#include <reticolo/orch/span.hpp>
#include <reticolo/orch/thread_plan.hpp>
#include <reticolo/updater/concepts.hpp>
#include <reticolo/updater/hmc/hmc.hpp>
#include <reticolo/updater/hmc/integrators.hpp>
// NOLINTEND(misc-include-cleaner)

// Short namespace aliases. One `using namespace reticolo;` per app then gives
// terse `act::Phi4`, `updater::Hmc`, `obs::reduce`, `io::Writer`, `cli::Parser`.
namespace reticolo {
namespace act = action;
}  // namespace reticolo
