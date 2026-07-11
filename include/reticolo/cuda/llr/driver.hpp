#pragma once

// Device LLR driver (host-callable, no kernels). The GPU analog of orch::llr::run,
// model B: replicas overlap because every replica's trajectory is enqueued on
// its own stream BEFORE any sync (the measurement loop is trajectory-outer /
// replica-inner, unlike the CPU driver's replica-outer omp loop).
//
// Reuses the CPU host math verbatim — orch::llr::nr_update / rm_update. Exchange is
// param-swap (see Replica::swap_window): windows migrate across slots, so E_n is
// written as a per-slot series and analysis groups by E_n. The acceptance is the
// same Gaussian-window form the CPU uses (linear tilt + window quadratic
// cross-term, see orch::llr::try_exchange); param-swap gives the identical acceptance
// and a slot-relabeled state, so the ensemble matches the CPU config-swap up to
// the (E_n-grouped) relabeling.

#include <reticolo/core/log.hpp>
#include <reticolo/io/writer.hpp>
#include <reticolo/orch/llr/update_a.hpp>

#include <cmath>
#include <cstddef>
#include <format>
#include <memory>
#include <vector>

namespace reticolo::cuda::llr {

// Overlapped warm-in: advance all not-yet-converged replicas one batch at a time,
// launching every replica's batch (async) before checking any, so the batches
// overlap across streams. Physically identical to per-replica serial warm — each
// replica runs the same trajectories until its own |S−E_n| < threshold·δ — only
// the scheduling changes.
template <class Replica>
void warm_all(std::vector<std::unique_ptr<Replica>>& reps,
              int max_batches,
              int batch               = 10,
              double threshold_sigmas = 1.0) {
    std::size_t const n_rep = reps.size();
    std::vector<char> done(n_rep, 0);
    for (int b = 0; b < max_batches; ++b) {
        for (std::size_t n = 0; n < n_rep; ++n) {
            if (done[n] == 0) {
                reps[n]->warm_launch(batch);  // async → overlap
            }
        }
        std::size_t remaining = 0;
        for (std::size_t n = 0; n < n_rep; ++n) {
            if (done[n] != 0) {
                continue;
            }
            if (reps[n]->warm_reached(threshold_sigmas)) {
                done[n] = 1;
            } else {
                ++remaining;
            }
        }
        if (remaining == 0) {
            break;
        }
    }
}

struct DriverSpec {
    int n_nr;
    int n_therm_nr;
    int n_meas_nr;
    int n_rm;
    int n_therm_rm;
    int n_meas_rm;
    double delta;
    double e_min;
    // NOLINTNEXTLINE(readability-identifier-naming) physics convention
    double E_max;  // already snapped to the (n_rep - 1) * delta grid
    double d_e;
    bool exchange = true;
};

template <class Replica, class ExchRng>
void run(std::vector<std::unique_ptr<Replica>>& reps,
         ExchRng& exch_rng,
         DriverSpec const& spec,
         io::Writer& out) {
    int const n_rep           = static_cast<int>(reps.size());
    std::size_t const n_rep_u = reps.size();

    out.attr<int>("/cfg@n_rep", n_rep);
    out.attr<int>("/cfg@n_nr", spec.n_nr);
    out.attr<int>("/cfg@n_rm", spec.n_rm);
    out.attr<double>("/cfg@delta", spec.delta);
    out.attr<double>("/cfg@E_min", spec.e_min);
    out.attr<double>("/cfg@E_max", spec.E_max);
    out.attr<double>("/cfg@dE", spec.d_e);

    auto e_n0_series = out.series<double>("/cfg/E_n");  // initial per-slot centres
    for (auto const& r : reps) {
        e_n0_series.append(r->e_n());
    }

    std::vector<io::Series<double>> a_series;
    std::vector<io::Series<double>> de_series;
    std::vector<io::Series<double>> en_series;  // per-slot E_n (time-varying: param-swap)
    a_series.reserve(n_rep_u);
    de_series.reserve(n_rep_u);
    en_series.reserve(n_rep_u);
    for (int n = 0; n < n_rep; ++n) {
        a_series.emplace_back(out.series<double>(std::format("/replica_{:03d}/a", n)));
        de_series.emplace_back(out.series<double>(std::format("/replica_{:03d}/dE", n)));
        en_series.emplace_back(out.series<double>(std::format("/replica_{:03d}/E_n", n)));
    }
    auto exch_series = out.series<int>("/exchange/accepted");

    // Host-free overlapped measurement: the entire block (thermalise + n_meas
    // trajectories, every replica) is enqueued on the per-replica streams before
    // any sync, with each trajectory's ⟨S−E_n⟩ contribution accumulated ON the
    // device. Only one readback per replica at the end — no per-trajectory host
    // barrier. Returns ⟨dE⟩ per slot.
    auto measure = [&](int n_therm, int n_meas) {
        for (auto& r : reps) {
            r->thermalize(n_therm);  // async
            r->begin_measure();      // zero device accumulator (async)
        }
        for (int t = 0; t < n_meas; ++t) {
            for (auto& r : reps) {
                r->measure_trajectory();  // trajectory + on-device accumulate, async
            }
        }
        std::vector<double> de(n_rep_u);
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            de[n] = reps[n]->end_measure(n_meas);  // one readback per replica
        }
        return de;
    };

    auto drain = [&](std::vector<double> const& de) {
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            a_series[n].append(reps[n]->a());
            de_series[n].append(de[n]);
            en_series[n].append(reps[n]->e_n());
        }
    };

    log::info("llr", "NR phase  {} iters × {} replicas", spec.n_nr, n_rep_u);
    for (int k = 0; k < spec.n_nr; ++k) {
        auto const de = measure(spec.n_therm_nr, spec.n_meas_nr);
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            reps[n]->set_a(reticolo::orch::llr::nr_update(reps[n]->a(), de[n], spec.delta));
        }
        drain(de);
        log::info("llr", "NR iter  {:>3}/{}  done", k + 1, spec.n_nr);
    }

    log::info("llr", "RM phase  {} iters × {} replicas", spec.n_rm, n_rep_u);
    for (int s = 0; s < spec.n_rm; ++s) {
        auto const de = measure(spec.n_therm_rm, spec.n_meas_rm);
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            reps[n]->set_a(reticolo::orch::llr::rm_update(reps[n]->a(), de[n], spec.delta, s));
        }
        drain(de);

        int accepted = 0;
        int attempts = 0;
        if (spec.exchange) {
            std::size_t const off = static_cast<std::size_t>(s & 1);
            for (std::size_t i = off; i + 1 < reps.size(); i += 2) {
                ++attempts;
                double const e_i = reps[i]->energy();
                double const e_j = reps[i + 1]->energy();
                // Gaussian-window acceptance: linear tilt + the window quadratic
                // cross-term (exact for the Gaussian window; vanishes only when
                // the two windows share E_n and δ). Mirrors CPU orch::llr::try_exchange.
                double const dq   = e_i - e_j;
                double const qsum = e_i + e_j;
                double const d_i  = reps[i]->delta();
                double const d_j  = reps[i + 1]->delta();
                double const lin  = (reps[i]->a() - reps[i + 1]->a()) * dq;
                double const win =
                    (dq / 2.0) * (((qsum - (2.0 * reps[i]->e_n())) / (d_i * d_i)) -
                                  ((qsum - (2.0 * reps[i + 1]->e_n())) / (d_j * d_j)));
                double const log_p = lin + win;
                if (log_p >= 0.0 || exch_rng.uniform() < std::exp(log_p)) {
                    reps[i]->swap_window(*reps[i + 1]);
                    ++accepted;
                }
            }
        }
        exch_series.append(accepted);
        log::info("exch", "step  {:>3}  accepted  {}/{}", s + 1, accepted, attempts);
        log::info("llr", "RM iter  {:>3}/{}  done", s + 1, spec.n_rm);
    }
}

}  // namespace reticolo::cuda::llr
