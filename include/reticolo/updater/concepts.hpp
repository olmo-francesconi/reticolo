#pragma once

#include <reticolo/core/rng/rng.hpp>
#include <reticolo/core/rng/stream_set.hpp>

#include <concepts>

namespace reticolo::updater {

// What an updater needs of a generator: a usable Rng that can also be split into
// the independent per-slab streams a StreamSet is built from. Named once here
// because both updaters require exactly this, and because spelling it inline
// forces a template parameter called `R` — the whole point of naming it is that
// the parameter can then be called `Rng` like everywhere else, without shadowing
// the `reticolo::Rng` concept at the point of use (which GCC and clang do not
// even agree on).
template <class G>
concept SamplerRng = reticolo::Rng<G> && (JumpStream<G> || KeyedStream<G>);

// The contract every update algorithm satisfies — the surface that apps and the
// orchestration layer (orch::llr::Replica, orch::span::Chain) depend on. It is
// the updater-level analogue of action::HmcAction: duck-typed, no base class,
// checked at the use site. `updater::Hmc` is the sole implementation today; a new
// updater is a class modelling this concept (reusing the same integrator
// type-parameters), and the orchestrators pick it up unchanged.
//
// The three members are exactly what a driver needs from a sampler:
//   - step()          run one update, returning a result with `acceptance()`
//   - last_s_full()   S of the current config after the most recent update
//                     (so a driver reads the action without a fresh O(V) sweep)
//   - rng()           the owned RNG state, for checkpoint / resume
//
// `acceptance()` — a rate in [0,1] — is the blessed cross-updater accessor, and
// deliberately the ONLY thing the concept knows about the outcome. One HMC
// trajectory is accepted or not (`HmcResult::acceptance()` is 0 or 1); one
// Metropolis sweep is V independent accept tests and has no meaningful boolean.
// Algorithm-specific fields stay on the concrete result — `HmcResult::dH` /
// `::accepted`, `MetropolisResult::n_accepted` — where a driver that knows which
// sampler it holds can read them, and a generic one cannot accidentally depend
// on them.
template <class U>
concept Updater = requires(U& u, U const& cu) {
    { u.step().acceptance() } -> std::convertible_to<double>;
    { cu.last_s_full() } -> std::convertible_to<double>;
    u.rng();
};

}  // namespace reticolo::updater
