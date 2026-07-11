#pragma once

#include <concepts>
#include <string_view>

namespace reticolo::orch {

// Orchestration concepts — the contract the concurrent-execution spine relies
// on. The orchestration layer owns a set of WORKERS, each a self-contained
// simulation unit that does the actual sampling; the spine only runs them
// concurrently, drains their output serially, and (optionally) checkpoints
// them. It never mentions physics — no window, no tilt, no exchange.
//
// A Worker is deliberately minimal: it just has to name itself. That is enough
// for `orch::parallel_workers` to log/scope per worker. The heavier surface
// (a config buffer + owned RNG) is an opt-in refinement used only by the
// ensemble checkpoint.

template <class W>
concept Worker = requires(W const& w) {
    { w.id() } -> std::convertible_to<std::string_view>;
};

// Refinement: a worker whose full state can be snapshot to / restored from a
// checkpoint. `field()` is the config buffer (a Lattice / LinkLattice) and
// `rng()` the owned StreamSet — the two pieces a deterministic resume needs.
// Any per-worker payload beyond those (e.g. an LLR replica's tilt `a`) is
// written through the optional `save_extra` / `load_extra` hooks in
// orch/checkpoint.hpp, so this concept stays physics-free.
template <class W>
concept Checkpointable = Worker<W> && requires(W& w) {
    { w.field() };
    { w.rng() };
};

}  // namespace reticolo::orch
