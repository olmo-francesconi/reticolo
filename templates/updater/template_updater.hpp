#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// TEMPLATE — a new update algorithm (Langevin / heat-bath / overrelaxation…).
// COPY, don't edit here.
//
//   1. Copy to  include/reticolo/updater/<name>/<name>.hpp  (rename MyUpdater→…).
//   2. Fill every `// FILL IN` section — the step() body is the whole job.
//   3. Delete the `#error` line below.
//   4. Register: add `#include <reticolo/updater/<name>/<name>.hpp>` to
//      include/reticolo/reticolo.hpp. There is NO registry — apps instantiate it
//      directly (`updater::MyUpdater<Action, Rng>`), and BOTH orchestrators
//      (orch::span, orch::llr) pick it up unchanged because they depend only on
//      the `updater::Updater` concept, not on Hmc.
//   5. A `static_assert(updater::Updater<MyUpdater<…>>)` in a test (or at the
//      app use site) proves it models the contract.
//
// The contract (updater/concepts.hpp): step() → {dH, accepted}, last_s_full(),
// rng(). No base class. Own your randomness the same way Hmc does — take the
// family generator BY VALUE in the ctor and build a StreamSet — so chains are
// reproducible and checkpoint/resume works through rng().
//
// Mirrors include/reticolo/updater/hmc/hmc.hpp (the sole existing updater). A
// molecular-dynamics-style updater also takes an `Integrator` type-parameter
// like Hmc; a heat-bath / Langevin one does not — drop it if unused.
// ═══════════════════════════════════════════════════════════════════════════

#error "template: fill in the FILL IN sections, then delete this #error line."

#include <reticolo/action/concepts.hpp>
#include <reticolo/core/field_traits.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/rng/stream_set.hpp>
#include <reticolo/updater/concepts.hpp>

namespace reticolo::updater {

// Tuning knobs for one trajectory.
struct MyUpdaterSpec {
    // ── FILL IN ① — your per-trajectory parameters ───────────────────────────
    //   Worked example: a Metropolis-style update needs a proposal width; a
    //   heat-bath needs none; an MD updater needs τ + n_md (see HmcSpec).
    double step_size = 0.1;
    int n_sweeps     = 1;
    // If your update THREADS the lattice, also mirror HmcSpec.n_threads /
    // slabs_per_thread here and bind them with exec::team_scope / slab_scope in
    // step() — otherwise leave this serial and ignore threading.
    // ──────────────────────────────────────────────────────────────────────────
};

// The step() return type — must expose `.dH` and `.accepted` (the concept).
// Reuse updater::HmcResult if it fits; otherwise a local struct like this.
struct MyUpdaterResult {
    double dH     = 0.0;  // NOLINT(readability-identifier-naming) physics: ΔH
    bool accepted = false;
};

template <class Action,
          class Rng,
          class Field = Lattice<typename Action::value_type>,
          class T     = typename Action::value_type>
class MyUpdater {
public:
    using value_type = T;
    using field_type = Field;

    // The action + field are held by reference (the caller owns them). The RNG is
    // taken BY VALUE and turned into an owned StreamSet: `streams_.driver()` is
    // one serial stream (Metropolis accept, global proposals); the site streams
    // are for per-site PARALLEL draws (pass n_streams = one per canonical slab,
    // like Hmc, only if you thread — a serial updater uses just the driver).
    // NEVER pass the same rng lvalue to two updaters: copies give identical
    // chains; derive distinct seeds instead.
    MyUpdater(Action const& action,
              Field& field,
              Rng rng,
              MyUpdaterSpec spec,
              log::Mode announce = log::Mode::normal)
        : action_{action}, field_{field}, spec_{spec}, streams_{std::move(rng), /*n_streams=*/1} {
        if (announce == log::Mode::normal) {
            log::algo(*this);
        }
    }

    void describe(log::Entry& e) const {
        e.line("MyUpdater<{}>", scalar_name<T>());
        // ── FILL IN ② — one e.param() per knob (shown in the run banner) ──────
        //   Worked example:
        //       e.param("step={:.3g}", spec_.step_size);
        //       e.param("sweeps={}", spec_.n_sweeps);
        e.param("step={:.3g}", spec_.step_size);
        // ──────────────────────────────────────────────────────────────────────
    }

    // Run ONE trajectory and return {dH, accepted}. THIS is your algorithm.
    // What you have to work with:
    //   action_.s_full(field_)               → total S (double)
    //   action_.compute_force(field_, force) → writes −dS/dφ into a Lattice force
    //   field_.data() / field_.nsites()      → raw T* + site count (scalar field)
    //   streams_.driver()                    → serial Rng: .uniform() ∈[0,1),
    //                                          .normal() ~N(0,1), .uniform_int(n)
    MyUpdaterResult step(log::Mode log_mode = log::Mode::normal) {
        (void)log_mode;
        // ── FILL IN ③ — one trajectory ───────────────────────────────────────
        // Worked example — a GLOBAL Metropolis update (Gaussian-shift proposal).
        // Swap step 2 (the proposal) for your algorithm; the accept/reject +
        // bookkeeping shape is the same for any Metropolis-type updater:
        //
        //     auto& drv       = streams_.driver();
        //     double const s0 = action_.s_full(field_);        // S before
        //     T* const data          = field_.data();          // 1. snapshot
        //     std::size_t const n    = field_.nsites();
        //     std::vector<T> saved(data, data + n);
        //     for (std::size_t k = 0; k < n; ++k)               // 2. propose
        //         data[k] += static_cast<T>(drv.normal() * spec_.step_size);
        //     double const s1 = action_.s_full(field_);         // S after
        //     double const dS = s1 - s0;                        // 3. accept/reject
        //     bool const accepted = (dS <= 0.0) || (drv.uniform() < std::exp(-dS));
        //     if (!accepted)
        //         std::copy(saved.begin(), saved.end(), data);  //    undo on reject
        //     last_s_full_ = accepted ? s1 : s0;                // 4. S of the END config
        //     return {.dH = dS, .accepted = accepted};          // 5. report
        //
        // (A heat-bath / overrelaxation updater replaces the propose+accept with
        //  its exact per-site draw and returns accepted = true.)
        MyUpdaterResult result{};
        last_s_full_ = action_.s_full(field_);
        return result;  // ← replace with your trajectory
        // ──────────────────────────────────────────────────────────────────────
    }

    // S of the current config after the most recent step() — no fresh O(V) sweep.
    [[nodiscard]] double last_s_full() const noexcept { return last_s_full_; }

    // The owned randomness, for checkpoint / resume (io::save_config/load_config).
    [[nodiscard]] StreamSet<Rng>& rng() noexcept { return streams_; }
    [[nodiscard]] StreamSet<Rng> const& rng() const noexcept { return streams_; }

private:
    static_assert(action::HmcAction<Action, Field>,
                  "the action must model action::HmcAction (s_full + compute_force)");

    Action const& action_;
    Field& field_;
    MyUpdaterSpec spec_;
    StreamSet<Rng> streams_;
    double last_s_full_ = 0.0;
};

}  // namespace reticolo::updater
