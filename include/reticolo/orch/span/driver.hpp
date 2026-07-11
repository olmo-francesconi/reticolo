#pragma once

#include <reticolo/core/log.hpp>
#include <reticolo/io/writer.hpp>
#include <reticolo/orch/ensemble.hpp>
#include <reticolo/orch/thread_plan.hpp>

#include <cstddef>
#include <format>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace reticolo::orch::span {

// Parameter-span orchestrator: run every worker's HMC chain concurrently through
// the generic orch spine, recording per-worker time series. A two-phase object
// mirroring orch::llr::Orchestrator — setup() opens the /worker_NNN series and
// announces the ensemble, run() drives the concurrent waves + serial drain.
//
// Unlike LLR (whose replica ladder is a uniform E_n sweep the orchestrator can
// synthesise), span workers carry an arbitrary parameter grid — that grid is the
// app's whole reason to exist — so the app builds the workers and hands them over
// by move. The schedule (thermalise, then measure) is identical across workers,
// so it lives here once.

struct Schedule {
    int n_therm    = 0;
    int n_prod     = 0;
    int meas_every = 1;
};

// A named per-config observable recorded under /worker_NNN/obs/<name>. Evaluated
// once per measurement per worker in the serial drain (not a hot path), so a
// std::function closure over the field is fine.
template <class Field>
struct Observable {
    std::string name;
    std::function<double(Field const&)> measure;
};

template <class Chain>
class Orchestrator {
public:
    using field_type = Chain::field_type;

    Orchestrator(std::vector<std::unique_ptr<Chain>> workers,
                 ThreadPlan plan,
                 std::vector<Observable<field_type>> obs = {})
        : workers_{std::move(workers)}, plan_{plan}, obs_{std::move(obs)} {}

    // Open the per-worker series (stats + S_full + the app's observables) and
    // announce the ensemble. The writer phase is the app's call (so app-specific
    // /cfg attrs land at the right path).
    void setup(io::Writer& out) {
        std::size_t const n_w = workers_.size();
        d_h_.reserve(n_w);
        accepted_.reserve(n_w);
        s_prod_.reserve(n_w);
        obs_series_.resize(obs_.size());
        for (auto& os : obs_series_) {
            os.reserve(n_w);
        }
        for (std::size_t n = 0; n < n_w; ++n) {
            auto const g = std::format("/worker_{:03d}", n);
            d_h_.emplace_back(out.series<double>(g + "/stats/dH"));
            accepted_.emplace_back(out.series<int>(g + "/stats/accepted"));
            s_prod_.emplace_back(out.series<double>(g + "/obs/s"));
            for (std::size_t k = 0; k < obs_.size(); ++k) {
                obs_series_[k].emplace_back(out.series<double>(g + "/obs/" + obs_[k].name));
            }
        }

        log::info("span", "ensemble  {} workers", n_w);
        if (!workers_.empty()) {
            workers_.front()->announce_sampler();
        }
        log::info("span", "threads   m={} × {} concurrent", plan_.m, plan_.concurrency);
    }

    void run(Schedule const& sched) {
        std::size_t const n_w = workers_.size();
        log::info("span",
                  "schedule  therm {} · prod {} (meas every {})",
                  sched.n_therm,
                  sched.n_prod,
                  sched.meas_every);

        orch::parallel_workers(
            workers_, plan_, [&](std::size_t /*i*/, Chain& c) { c.thermalize(sched.n_therm); });

        for (int i = 0; i < sched.n_prod; ++i) {
            bool const measure = (i % sched.meas_every) == 0;
            orch::parallel_workers(
                workers_, plan_, [&](std::size_t /*n*/, Chain& c) { c.advance(); });
            // Serial drain — HDF5 writes are not thread-safe. Observables read the
            // worker's post-trajectory field directly (unchanged during the drain).
            for (std::size_t n = 0; n < n_w; ++n) {
                auto const& c = *workers_[n];
                d_h_[n].append(c.last_dh());
                accepted_[n].append(c.last_accepted() ? 1 : 0);
                if (measure) {
                    s_prod_[n].append(static_cast<double>(c.action().s_full(c.field())));
                    for (std::size_t k = 0; k < obs_.size(); ++k) {
                        obs_series_[k][n].append(obs_[k].measure(c.field()));
                    }
                }
            }
        }
        log::info("span", "done   {} workers × {} prod trajectories", n_w, sched.n_prod);
    }

    [[nodiscard]] std::vector<std::unique_ptr<Chain>>& workers() noexcept { return workers_; }

private:
    std::vector<std::unique_ptr<Chain>> workers_;
    ThreadPlan plan_;
    std::vector<Observable<field_type>> obs_;

    std::vector<io::Series<double>> d_h_;
    std::vector<io::Series<int>> accepted_;
    std::vector<io::Series<double>> s_prod_;
    std::vector<std::vector<io::Series<double>>> obs_series_;
};

}  // namespace reticolo::orch::span
