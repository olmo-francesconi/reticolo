#pragma once

#include <reticolo/orch/checkpoint.hpp>

#include <filesystem>
#include <memory>
#include <vector>

namespace reticolo::orch::llr {

// LLR ensemble checkpoint = the generic orch snapshot (orch/checkpoint.hpp) plus
// the two LLR-specific payloads it doesn't know about: each replica's tilt `a`
// (written per replica by Replica::save_extra / load_extra) and the shared
// exchange RNG (written here as the orchestrator-level extra). The /replica_NNN
// group prefix + /orch@* header layout are unchanged from before the spine
// migration, so old checkpoints resume identically.
//
// The threading split (ThreadPlan / plan_threads) and the schedule position
// (OrchState) now live in the generic spine — orch/thread_plan.hpp and
// orch/checkpoint.hpp. `OrchState` is re-exported here so the LLR apps keep
// naming it through the llr namespace.
using orch::OrchState;

template <class Replica, class ExchRng>
void save_ensemble(std::filesystem::path const& path,
                   std::vector<std::unique_ptr<Replica>>& reps,
                   ExchRng const& exch_rng,
                   OrchState const& state) {
    orch::save_ensemble(path, reps, state, "replica", [&](io::Writer& w) {
        w.rng_state("/orch/exch_rng", exch_rng);
    });
}

template <class Replica, class ExchRng>
[[nodiscard]] OrchState load_ensemble(std::filesystem::path const& path,
                                      std::vector<std::unique_ptr<Replica>>& reps,
                                      ExchRng& exch_rng) {
    return orch::load_ensemble(
        path, reps, "replica", [&](io::Reader& r) { exch_rng = r.rng_state("/orch/exch_rng"); });
}

}  // namespace reticolo::orch::llr
