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

// Parameter-span driver: run every worker's HMC chain concurrently through the
// generic orch spine, recording per-worker time series. The workers differ only
// in the action parameters the app baked into each; the schedule (thermalise,
// then measure) is identical across them, so it lives here once.
//
// The app owns worker construction (it built the parameter grid) and stamps any
// /cfg metadata (e.g. the swept values) before calling run — mirroring how an
// LLR app stamps its own attrs. run owns the mechanics: the concurrent waves,
// the /worker_NNN series, and the serial drain.

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
void run(std::vector<std::unique_ptr<Chain>>& workers,
         Schedule const& sched,
         ThreadPlan const& plan,
         std::vector<Observable<typename Chain::field_type>> const& obs,
         io::Writer& out) {
    std::size_t const n_w = workers.size();

    // Per-worker series: stats (always) + S_full (universal to any HmcAction) +
    // the app's extra observables.
    std::vector<io::Series<double>> d_h;
    std::vector<io::Series<int>> accepted;
    std::vector<io::Series<double>> s_prod;
    d_h.reserve(n_w);
    accepted.reserve(n_w);
    s_prod.reserve(n_w);
    std::vector<std::vector<io::Series<double>>> obs_series(obs.size());
    for (auto& os : obs_series) {
        os.reserve(n_w);
    }
    for (std::size_t n = 0; n < n_w; ++n) {
        auto const g = std::format("/worker_{:03d}", n);
        d_h.emplace_back(out.series<double>(g + "/stats/dH"));
        accepted.emplace_back(out.series<int>(g + "/stats/accepted"));
        s_prod.emplace_back(out.series<double>(g + "/obs/s"));
        for (std::size_t k = 0; k < obs.size(); ++k) {
            obs_series[k].emplace_back(out.series<double>(g + "/obs/" + obs[k].name));
        }
    }

    log::info("span", "ensemble  {} workers", n_w);
    if (!workers.empty()) {
        workers.front()->announce_sampler();
    }
    log::info("span",
              "schedule  therm {} · prod {} (meas every {})",
              sched.n_therm,
              sched.n_prod,
              sched.meas_every);
    log::info("span", "threads   m={} × {} concurrent", plan.m, plan.concurrency);

    orch::parallel_workers(
        workers, plan, [&](std::size_t /*i*/, Chain& c) { c.thermalize(sched.n_therm); });

    for (int i = 0; i < sched.n_prod; ++i) {
        bool const measure = (i % sched.meas_every) == 0;
        orch::parallel_workers(workers, plan, [&](std::size_t /*n*/, Chain& c) { c.advance(); });
        // Serial drain — HDF5 writes are not thread-safe. Observables read the
        // worker's post-trajectory field directly (unchanged during the drain).
        for (std::size_t n = 0; n < n_w; ++n) {
            auto const& c = *workers[n];
            d_h[n].append(c.last_dh());
            accepted[n].append(c.last_accepted() ? 1 : 0);
            if (measure) {
                s_prod[n].append(static_cast<double>(c.action().s_full(c.field())));
                for (std::size_t k = 0; k < obs.size(); ++k) {
                    obs_series[k][n].append(obs[k].measure(c.field()));
                }
            }
        }
    }
    log::info("span", "done   {} workers × {} prod trajectories", n_w, sched.n_prod);
}

}  // namespace reticolo::orch::span
