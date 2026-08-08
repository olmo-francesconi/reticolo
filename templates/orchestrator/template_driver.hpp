#pragma once

// ═══════════════════════════════════════════════════════════════════════════
// TEMPLATE — an orchestrator (the concurrent driver).  COPY, don't edit here.
//
//   1. Copy to  include/reticolo/orch/<name>/driver.hpp (rename MyOrchestrator→…).
//   2. Fill every `// FILL IN` section.
//   3. Delete the `#error` line below.
//   4. Register: create include/reticolo/orch/<name>.hpp that #includes this and
//      worker.hpp, then #include <reticolo/orch/<name>.hpp> in reticolo.hpp.
//
// A two-phase object over the physics-free spine: setup(out) wires up IO (opens
// the per-worker series, logs the ensemble); run(...) drives the schedule via
// `orch::parallel_workers` (the ONE concurrent primitive) and drains output
// SERIALLY (HDF5 writes are not thread-safe — the orchestrator owns its loop).
// The app builds the workers (it owns the parameter grid / geometry) and hands
// them over by move, then stamps any /cfg attrs before setup().
//
// Depends only on `orch::Worker` + the spine — never on a specific action or
// updater. Mirrors include/reticolo/orch/span/driver.hpp (the minimal example);
// orch/llr/orchestrator.hpp shows the richer version (build ladder in setup,
// serial coupling/exchange between waves, checkpoint/resume).
// ═══════════════════════════════════════════════════════════════════════════

#error "template: fill in the FILL IN sections, then delete this #error line."

#include <reticolo/core/log.hpp>
#include <reticolo/io/writer.hpp>
#include <reticolo/orch/ensemble.hpp>
#include <reticolo/orch/thread_plan.hpp>

#include <cstddef>
#include <format>
#include <memory>
#include <vector>

namespace reticolo::orch::myorch {  // FILL IN ⓪: your orchestrator's namespace

// Per-trajectory schedule. Add yours.
struct Schedule {
    int n_therm = 0;
    int n_prod  = 0;
};

// PARAMETER SHAPE — keep it: the same words in the same order as both shipped
// orchestrators, with `Spec` NESTED so it is parameterised by exactly these
// arguments and the sampler knobs come from the tag (`Sampler::spec_type`) rather
// than being spelled out per sampler.
template <class Action, class Rng, class Sampler>
class MyOrchestrator {
public:
    using worker_type = MyWorker<Action, Rng, Sampler>;  // FILL IN: your worker

    // FILL IN ⓪ — whatever describes the ensemble. The rule of thumb: if the
    // orchestrator can DERIVE the per-worker setup from a few numbers (an LLR
    // ladder is E_n = E_min + n·dE), take the numbers; if the layout is arbitrary
    // (a parameter span), take the list of actions and derive everything else.
    struct Spec {
        std::vector<std::size_t> shape;
        unsigned long long seed = 0;
        Sampler::spec_type sampler{};  // {.tau,.n_md} or {.sigma} — never both
        int worker_threads = 1;        // ensemble-level, resolved into `sampler`
    };

    explicit MyOrchestrator(Spec spec) : spec_{std::move(spec)} {}

    // Plan threads, BUILD the workers, open the output series, announce. The app
    // starts the writer phase and stamps app-specific /cfg attrs before this.
    void setup(io::Writer& out) {
        std::size_t const n_w = /* FILL IN: how many workers this Spec describes */ 0;
        plan_                 = orch::plan_threads(static_cast<int>(n_w), spec_.worker_threads);

        auto sampler_spec = spec_.sampler;
        if constexpr (requires { sampler_spec.n_threads; }) {
            sampler_spec.n_threads = plan_.m;  // not every sampler has a team
        }
        workers_.reserve(n_w);
        {
            auto const quiet = log::quiet();  // silence per-worker ctor announces
            for (std::size_t n = 0; n < n_w; ++n) {
                workers_.push_back(
                    std::make_unique<worker_type>(std::format("w{:03d}", n),
                                                  spec_.shape, /* FILL IN: this worker's action */
                                                  Rng{spec_.seed + 1ULL + n},
                                                  sampler_spec));
            }
        }
        // ── FILL IN ① — open one io::Series per worker per recorded quantity ──
        //   Series are opened ONCE here and appended each trajectory in run()'s
        //   drain; they flush + close on destruction. Group them under /worker_NNN.
        //   Worked example (acceptance + S per worker):
        //       acc_.reserve(n_w); s_.reserve(n_w);
        //       for (std::size_t n = 0; n < n_w; ++n) {
        //           auto const g = std::format("/worker_{:03d}", n);
        //           acc_.emplace_back(out.series<double>(g + "/stats/acceptance"));
        //           s_.emplace_back(out.series<double>(g + "/obs/s"));
        //       }
        //   A sampler-generic driver records `acceptance()` — the only outcome
        //   `updater::Updater` guarantees. If you want HMC's ΔH / accept flag too,
        //   gate those series on a `requires` (see orch::span::driver's
        //   `k_hmc_stats`) so a local-updater run does not open empty datasets.
        acc_.reserve(n_w);
        for (std::size_t n = 0; n < n_w; ++n) {
            auto const g = std::format("/worker_{:03d}", n);
            acc_.emplace_back(out.series<double>(g + "/stats/acceptance"));
        }
        // ──────────────────────────────────────────────────────────────────────

        log::info("myorch", "ensemble  {} workers", n_w);
        if (!workers_.empty()) {
            workers_.front()->announce_sampler();
        }
        log::info("myorch", "threads   m={} × {} concurrent", plan_.m, plan_.concurrency);
    }

    void run(Schedule const& sched) {
        std::size_t const n_w = workers_.size();

        // Wave: the heavy per-worker compute, all workers at once under the plan.
        // Keep it free of cross-worker coupling and IO (a foreign OpenMP region,
        // so a device backend maps the same body onto a stream fan-out).
        orch::parallel_workers(workers_, plan_, [&](std::size_t /*i*/, worker_type& w) {
            w.thermalize(sched.n_therm);
        });

        for (int i = 0; i < sched.n_prod; ++i) {
            orch::parallel_workers(
                workers_, plan_, [&](std::size_t /*n*/, worker_type& w) { w.advance(); });
            // Serial drain — HDF5 is not thread-safe. Read each worker's stashed
            // post-trajectory results and append. Any cross-worker coupling
            // (e.g. LLR's replica exchange) also goes HERE, between waves.
            for (std::size_t n = 0; n < n_w; ++n) {
                // ── FILL IN ② — append this trajectory's stashed outputs ──────
                //   Worked example (matching the series opened in FILL IN ①):
                //       acc_[n].append(workers_[n]->last_acceptance());
                //       s_[n].append(workers_[n]->action().s_full(workers_[n]->field()));
                //   `acceptance()` is the only outcome `updater::Updater`
                //   guarantees, so it is what a sampler-generic driver records.
                //   For algorithm-specific extras (HMC's dH) gate the series on a
                //   `requires` — see orch::span::driver's `k_hmc_stats`.
                acc_[n].append(workers_[n]->last_acceptance());
                // ──────────────────────────────────────────────────────────────
            }
        }
        log::info("myorch", "done   {} workers × {} prod", n_w, sched.n_prod);
    }

    // Exposed so the app can do worker-specific init between setup() and run()
    // (e.g. hot-start / cold-start the fields), like the LLR apps do.
    // App-specific per-worker init (a gauge cold start) goes here, between
    // setup() and run() — the seam both shipped orchestrators offer.
    [[nodiscard]] std::vector<std::unique_ptr<worker_type>>& workers() noexcept { return workers_; }

private:
    Spec spec_;
    std::vector<std::unique_ptr<worker_type>> workers_;
    ThreadPlan plan_{};

    // ── FILL IN ③ — one series vector per quantity (match FILL IN ① and ②) ────
    //   Worked example:
    //       std::vector<io::Series<double>> acc_, s_;
    std::vector<io::Series<double>> acc_;
    // ──────────────────────────────────────────────────────────────────────────
};

}  // namespace reticolo::orch::myorch
