#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// TEMPLATE — an orchestration WORKER.  COPY, don't edit here.
//
//   1. Copy to  include/reticolo/orch/<name>/worker.hpp  (rename MyWorker → …).
//   2. Fill every `// FILL IN` section.
//   3. Delete the `#error` line below.
//   4. Aggregated by orch/<name>.hpp together with the driver.
//
// A Worker is one self-contained simulation unit — it owns a field + an updater
// and does the actual sampling; the driver only runs many of them concurrently
// and drains their output. It must model `orch::Worker` (just `id()`), and to be
// checkpointable also `orch::Checkpointable` (`field()` + `rng()`). Any extra
// per-worker state that needs saving (e.g. LLR's tilt `a`) goes through optional
// `save_extra`/`load_extra` hooks (see orch/llr/replica.hpp).
//
// Non-movable: the updater holds references into this worker's own members, so
// keep workers in `std::vector<std::unique_ptr<MyWorker<…>>>`.
//
// PARAMETER SHAPE — keep it. Every action-carrying template in the library reads
// `<Action, Rng, Sampler, …>`, in that order, with the same words:
//   * the element and field types are NOT parameters — the action names both
//     (`value_type` / `field_type`), so gauge and scalar workers read alike;
//   * the MD integrator is NOT a parameter either — it rides on the sampler tag
//     (`updater::HmcSampler<updater::integ::Omelyan4>`), so it does not exist
//     when the sampler is a local updater;
//   * the sampler has no default: name the updater at the call site.
//
// Mirrors include/reticolo/orch/span/worker.hpp (the minimal example).
// ═══════════════════════════════════════════════════════════════════════════

#error "template: fill in the FILL IN sections, then delete this #error line."

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/log_helpers.hpp>
#include <reticolo/updater/concepts.hpp>
#include <reticolo/updater/samplers.hpp>

#include <string>
#include <string_view>
#include <utility>

namespace reticolo::orch::myorch {  // FILL IN ⓪: your orchestrator's namespace

template <class Action, class Rng, class Sampler>
class MyWorker {
public:
    using value_type        = Action::value_type;
    using field_type        = Action::field_type;
    using action_type       = Action;
    using sampler_type      = Sampler::template type<Action, Rng>;
    using sampler_spec_type = Sampler::spec_type;

    static constexpr std::string_view log_tag = "wrkr";

    // Owns its field + a BY-VALUE copy of the action (so any per-action mutable
    // state stays per-worker — required for the parallel-over-workers loop) + its
    // updater (which references both). Add whatever else the worker needs.
    MyWorker(std::string id,
             field_type::SizeVec const& shape,
             Action action,
             Rng rng,
             sampler_spec_type const& spec)
        : id_{std::move(id)}, field_{shape}, action_{std::move(action)},
          sampler_{make_sampler_(action_, field_, std::move(rng), spec)} {}

    MyWorker(MyWorker const&)            = delete;
    MyWorker& operator=(MyWorker const&) = delete;
    MyWorker(MyWorker&&)                 = delete;
    MyWorker& operator=(MyWorker&&)      = delete;
    ~MyWorker()                          = default;

    // Announce the shared sampler once (the driver calls this on ONE worker).
    void announce_sampler() const { log::algo(sampler_); }

    // orch::Worker — the only hard requirement.
    [[nodiscard]] std::string_view id() const noexcept { return id_; }

    // ── FILL IN ① — the per-worker sim methods your driver's waves call ───────
    //   These are the CONTRACT with your driver (template_driver.hpp): each is
    //   invoked inside a `parallel_workers` wave (so ALL workers, concurrently —
    //   keep them pure per-worker compute, no cross-worker access, no HDF5). Any
    //   result the driver's SERIAL drain needs to record, stash in a member and
    //   expose with a getter (the drain reads it after the wave). Pick whatever
    //   methods your workflow needs; the two below are span::Chain's full set.
    //
    //   Worked example (span::Chain) — thermalise, then one measured trajectory:
    void thermalize(int n) {
        for (int i = 0; i < n; ++i) {
            (void)sampler_.step(log::Mode::silent);
        }
    }

    void advance() {
        // `acceptance()` is the only outcome `updater::Updater` guarantees, so a
        // sampler-generic worker stashes that. Algorithm-specific extras (dH) are
        // reachable behind a `requires` — see span::Chain's `k_hmc_stats`.
        last_acceptance_ = sampler_.step(log::Mode::silent).acceptance();
    }

    [[nodiscard]] double last_acceptance() const noexcept { return last_acceptance_; }
    //   (An LLR replica's richer set — thermalize / sample / warm_into_window /
    //    try_exchange — lives in orch/llr/replica.hpp if you need a fuller model.)
    // ──────────────────────────────────────────────────────────────────────────

    // orch::Checkpointable (drop these two if the worker needn't checkpoint).
    [[nodiscard]] field_type& field() noexcept { return field_; }
    [[nodiscard]] field_type const& field() const noexcept { return field_; }
    [[nodiscard]] StreamSet<Rng>& rng() noexcept { return sampler_.rng(); }

    [[nodiscard]] Action const& action() const noexcept { return action_; }

private:
    static_assert(updater::Updater<sampler_type>,
                  "the worker's sampler must model updater::Updater");

    // The tag's optional `integrator` alias decides whether the updater ctor
    // takes one; guaranteed elision constructs in place (updaters are not
    // movable — they hold references into this worker).
    [[nodiscard]] static sampler_type
    make_sampler_(Action& a, field_type& f, Rng rng, sampler_spec_type const& spec) {
        if constexpr (requires { typename Sampler::integrator; }) {
            return sampler_type{
                a, f, std::move(rng), spec, typename Sampler::integrator{}, log::Mode::silent};
        } else {
            return sampler_type{a, f, std::move(rng), spec, log::Mode::silent};
        }
    }

    std::string id_;
    field_type field_;
    Action action_;
    sampler_type sampler_;
    double last_acceptance_ = 0.0;
};

}  // namespace reticolo::orch::myorch
