#pragma once

#include <reticolo/core/log/log.hpp>
#include <reticolo/io/writer.hpp>
#include <reticolo/orch/ensemble.hpp>
#include <reticolo/orch/span/worker.hpp>
#include <reticolo/orch/thread_plan.hpp>

#include <cstddef>
#include <format>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace reticolo::orch::span {

// Parameter-span orchestrator: run many chains concurrently through the generic
// orch spine, recording per-worker time series. Two-phase, exactly like
// orch::llr::Orchestrator — setup() plans threads, builds the worker grid and
// opens the /worker_NNN series; run() drives the concurrent waves + serial drain.
//
// The one thing it cannot synthesise is the GRID. An LLR ladder is always
// E_n = E_min + n·dE, so three numbers describe it; a parameter span is
// arbitrary — log-spaced, hand-picked, multi-dimensional — and choosing it is the
// app's whole reason to exist. So `Spec::points` IS the grid: one action per
// worker, and everything else about worker construction (ids, per-worker seeds,
// the thread plan) is derived here rather than re-written in every app.
// `span::scan` below covers the common one-knob linear case in a line.

struct Schedule {
    int n_therm    = 0;
    int n_prod     = 0;
    int meas_every = 1;
};

// `n` copies of `base` with one scalar member swept linearly over [lo, hi] — the
// span analogue of an LLR ladder, and the usual way to fill `Spec::points`:
//
//     .points = span::scan(act::Phi4<double>{.lambda = lam},
//                          &act::Phi4<double>::kappa, k_min, k_max, n_workers)
//
// Anything this does not cover (log spacing, a 2-D grid, hand-picked values) is a
// plain loop that pushes into `points`; there is no second mechanism.
template <class Action, class T>
[[nodiscard]] inline std::vector<Action> scan(Action base, T Action::* knob, T lo, T hi, int n) {
    std::vector<Action> pts;
    auto const count = static_cast<std::size_t>(n < 1 ? 1 : n);
    pts.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        double const t = count == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(count - 1);
        Action a       = base;
        a.*knob        = static_cast<T>(lo + (t * (hi - lo)));
        pts.push_back(a);
    }
    return pts;
}

// A named per-config observable recorded under /worker_NNN/obs/<name>. Evaluated
// once per measurement per worker in the serial drain (not a hot path), so a
// std::function closure over the field is fine.
template <class Field>
struct Observable {
    std::string name;
    std::function<double(Field const&)> measure;
};

// Same parameter shape as orch::llr::Orchestrator; `Spec` is nested so it is
// parameterised by exactly these arguments and there is nothing extra to spell.
template <class Action, class Rng, class Sampler>
class Orchestrator {
public:
    using chain_type = Chain<Action, Rng, Sampler>;
    using field_type = chain_type::field_type;

    // The sampler's own knobs, whatever they are — (tau, n_md) for HMC, sigma for
    // a local updater — so a Spec never advertises a field the chosen sampler
    // ignores. `worker_threads` stays out here: it is the ENSEMBLE thread request
    // (how many to try per worker), which setup() resolves into the sampler spec.
    struct Spec {
        std::vector<std::size_t> shape;
        unsigned long long seed = 0;
        std::vector<Action> points;  // one worker per entry — THE grid
        Sampler::spec_type sampler{};
        int worker_threads = 1;  // → plan_threads(points.size(), worker_threads)
    };

    // Which per-update diagnostics this ensemble records — decided by the
    // worker's sampler, not by this driver. An HMC chain writes /stats/dH +
    // /stats/accepted; a local-update chain writes /stats/acceptance, because a
    // sweep is V independent accept tests with no meaningful boolean. Same split
    // the standalone apps make.
    static constexpr bool k_hmc_stats = chain_type::k_hmc_stats;

    Orchestrator(Spec spec, std::vector<Observable<field_type>> obs = {})
        : spec_{std::move(spec)}, obs_{std::move(obs)} {}

    // Plan threads, build the worker grid, open the per-worker series (stats +
    // S_full + the app's observables) and announce the ensemble. The writer phase
    // is the app's call, so app-specific /cfg attrs land at the right path.
    void setup(io::Writer& out) {
        std::size_t const n_w = spec_.points.empty() ? 0 : spec_.points.size();
        plan_                 = orch::plan_threads(static_cast<int>(n_w), spec_.worker_threads);

        // The resolved team size belongs to the sampler, but not every sampler has
        // to have one — probe rather than assume.
        auto sampler_spec = spec_.sampler;
        if constexpr (requires { sampler_spec.n_threads; }) {
            sampler_spec.n_threads = plan_.m;
        }

        workers_.reserve(n_w);
        {
            auto const quiet = log::quiet();  // silence per-worker ctor announces
            for (std::size_t n = 0; n < n_w; ++n) {
                workers_.push_back(std::make_unique<chain_type>(std::format("w{:03d}", n),
                                                                spec_.shape,
                                                                spec_.points[n],
                                                                Rng{spec_.seed + 1ULL + n},
                                                                sampler_spec));
            }
        }

        d_h_.reserve(n_w);
        accepted_.reserve(n_w);
        acceptance_.reserve(n_w);
        s_prod_.reserve(n_w);
        obs_series_.resize(obs_.size());
        for (auto& os : obs_series_) {
            os.reserve(n_w);
        }
        for (std::size_t n = 0; n < n_w; ++n) {
            auto const g = std::format("/worker_{:03d}", n);
            if constexpr (k_hmc_stats) {
                d_h_.emplace_back(out.series<double>(g + "/stats/dH"));
                accepted_.emplace_back(out.series<int>(g + "/stats/accepted"));
            } else {
                acceptance_.emplace_back(out.series<double>(g + "/stats/acceptance"));
            }
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

        orch::parallel_workers(workers_, plan_, [&](std::size_t /*i*/, chain_type& c) {
            c.thermalize(sched.n_therm);
        });

        for (int i = 0; i < sched.n_prod; ++i) {
            bool const measure = (i % sched.meas_every) == 0;
            orch::parallel_workers(
                workers_, plan_, [&](std::size_t /*n*/, chain_type& c) { c.advance(); });
            // Serial drain — HDF5 writes are not thread-safe. Observables read the
            // worker's post-trajectory field directly (unchanged during the drain).
            for (std::size_t n = 0; n < n_w; ++n) {
                auto const& c = *workers_[n];
                if constexpr (k_hmc_stats) {
                    d_h_[n].append(c.last_dh());
                    accepted_[n].append(c.last_accepted() ? 1 : 0);
                } else {
                    acceptance_[n].append(c.last_acceptance());
                }
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

    // App-specific per-worker init (a gauge cold start, a hot start) goes here,
    // between setup() and run() — the same seam orch::llr::Orchestrator offers.
    [[nodiscard]] std::vector<std::unique_ptr<chain_type>>& workers() noexcept { return workers_; }

private:
    Spec spec_;
    std::vector<Observable<field_type>> obs_;
    std::vector<std::unique_ptr<chain_type>> workers_;
    ThreadPlan plan_{};

    std::vector<io::Series<double>> d_h_;         // HMC samplers only
    std::vector<io::Series<int>> accepted_;       // HMC samplers only
    std::vector<io::Series<double>> acceptance_;  // local samplers only
    std::vector<io::Series<double>> s_prod_;
    std::vector<std::vector<io::Series<double>>> obs_series_;
};

}  // namespace reticolo::orch::span
