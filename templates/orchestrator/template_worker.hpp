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
// Mirrors include/reticolo/orch/span/worker.hpp (the minimal example).
// ═══════════════════════════════════════════════════════════════════════════

#error "template: fill in the FILL IN sections, then delete this #error line."

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/log_helpers.hpp>
#include <reticolo/updater/concepts.hpp>
#include <reticolo/updater/hmc/hmc.hpp>
#include <reticolo/updater/hmc/integrators.hpp>

#include <string>
#include <string_view>
#include <utility>

namespace reticolo::orch::myorch {  // FILL IN ⓪: your orchestrator's namespace

template <class Action,
          class Rng,
          class Integrator = updater::integ::Omelyan2,
          class T          = typename Action::value_type,
          class Field      = Lattice<T>>
class MyWorker {
public:
    using value_type  = T;
    using field_type  = Field;
    using action_type = Action;

    static constexpr std::string_view log_tag = "wrkr";

    // Owns its field + a BY-VALUE copy of the action (so any per-action mutable
    // state stays per-worker — required for the parallel-over-workers loop) + its
    // updater (which references both). Add whatever else the worker needs.
    MyWorker(std::string id,
             typename Field::SizeVec const& shape,
             Action action,
             Rng rng,
             updater::HmcSpec const& spec)
        : id_{std::move(id)}, field_{shape}, action_{std::move(action)},
          hmc_{action_, field_, std::move(rng), spec, {}, log::Mode::silent} {}

    MyWorker(MyWorker const&)            = delete;
    MyWorker& operator=(MyWorker const&) = delete;
    MyWorker(MyWorker&&)                 = delete;
    MyWorker& operator=(MyWorker&&)      = delete;
    ~MyWorker()                          = default;

    // Announce the shared sampler once (the driver calls this on ONE worker).
    void announce_sampler() const { log::algo(hmc_); }

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
            (void)hmc_.step(log::Mode::silent);
        }
    }

    void advance() {
        auto const st = hmc_.step(log::Mode::silent);
        last_dh_      = st.dH;  // stash for the driver's serial drain
        last_acc_     = st.accepted;
    }

    [[nodiscard]] double last_dh() const noexcept { return last_dh_; }
    [[nodiscard]] bool last_accepted() const noexcept { return last_acc_; }
    //   (An LLR replica's richer set — thermalize / sample / warm_into_window /
    //    try_exchange — lives in orch/llr/replica.hpp if you need a fuller model.)
    // ──────────────────────────────────────────────────────────────────────────

    // orch::Checkpointable (drop these two if the worker needn't checkpoint).
    [[nodiscard]] Field& field() noexcept { return field_; }
    [[nodiscard]] Field const& field() const noexcept { return field_; }
    [[nodiscard]] StreamSet<Rng>& rng() noexcept { return hmc_.rng(); }

    [[nodiscard]] Action const& action() const noexcept { return action_; }

private:
    static_assert(updater::Updater<updater::Hmc<Action, Rng, Integrator, Field, T>>,
                  "the worker's sampler must model updater::Updater");

    std::string id_;
    Field field_;
    Action action_;
    updater::Hmc<Action, Rng, Integrator, Field, T> hmc_;
    double last_dh_ = 0.0;
    bool last_acc_  = false;
};

}  // namespace reticolo::orch::myorch
