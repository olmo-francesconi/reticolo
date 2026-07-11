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

template <class Worker>
class MyOrchestrator {
public:
    // The app builds the workers (its parameter grid is its reason to exist) and
    // moves them in, along with the thread split (orch::plan_threads).
    MyOrchestrator(std::vector<std::unique_ptr<Worker>> workers, ThreadPlan plan)
        : workers_{std::move(workers)}, plan_{plan} {}

    // Open the output series + announce the ensemble. The app starts the writer
    // phase and stamps app-specific /cfg attrs before calling this.
    void setup(io::Writer& out) {
        std::size_t const n_w = workers_.size();
        // ── FILL IN ① — open one io::Series per worker per recorded quantity ──
        //   Series are opened ONCE here and appended each trajectory in run()'s
        //   drain; they flush + close on destruction. Group them under /worker_NNN.
        //   Worked example (span records ΔH, an accept flag and S per worker):
        //       d_h_.reserve(n_w); acc_.reserve(n_w); s_.reserve(n_w);
        //       for (std::size_t n = 0; n < n_w; ++n) {
        //           auto const g = std::format("/worker_{:03d}", n);
        //           d_h_.emplace_back(out.series<double>(g + "/stats/dH"));
        //           acc_.emplace_back(out.series<int>(g + "/stats/accepted"));
        //           s_.emplace_back(out.series<double>(g + "/obs/s"));
        //       }
        d_h_.reserve(n_w);
        for (std::size_t n = 0; n < n_w; ++n) {
            auto const g = std::format("/worker_{:03d}", n);
            d_h_.emplace_back(out.series<double>(g + "/stats/dH"));
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
        orch::parallel_workers(
            workers_, plan_, [&](std::size_t /*i*/, Worker& w) { w.thermalize(sched.n_therm); });

        for (int i = 0; i < sched.n_prod; ++i) {
            orch::parallel_workers(
                workers_, plan_, [&](std::size_t /*n*/, Worker& w) { w.advance(); });
            // Serial drain — HDF5 is not thread-safe. Read each worker's stashed
            // post-trajectory results and append. Any cross-worker coupling
            // (e.g. LLR's replica exchange) also goes HERE, between waves.
            for (std::size_t n = 0; n < n_w; ++n) {
                // ── FILL IN ② — append this trajectory's stashed outputs ──────
                //   Worked example (matching the series opened in FILL IN ①):
                //       d_h_[n].append(workers_[n]->last_dh());
                //       acc_[n].append(workers_[n]->last_accepted() ? 1 : 0);
                //       s_[n].append(workers_[n]->action().s_full(workers_[n]->field()));
                d_h_[n].append(workers_[n]->last_dh());
                // ──────────────────────────────────────────────────────────────
            }
        }
        log::info("myorch", "done   {} workers × {} prod", n_w, sched.n_prod);
    }

    // Exposed so the app can do worker-specific init between setup() and run()
    // (e.g. hot-start / cold-start the fields), like the LLR apps do.
    [[nodiscard]] std::vector<std::unique_ptr<Worker>>& workers() noexcept { return workers_; }

private:
    std::vector<std::unique_ptr<Worker>> workers_;
    ThreadPlan plan_;

    // ── FILL IN ③ — one series vector per quantity (match FILL IN ① and ②) ────
    //   Worked example:
    //       std::vector<io::Series<double>> d_h_, s_;
    //       std::vector<io::Series<int>>    acc_;
    std::vector<io::Series<double>> d_h_;
    // ──────────────────────────────────────────────────────────────────────────
};

}  // namespace reticolo::orch::myorch
