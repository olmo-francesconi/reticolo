#pragma once

#include <reticolo/io/writer.hpp>
#include <reticolo/llr/exchange.hpp>
#include <reticolo/llr/log.hpp>
#include <reticolo/llr/update_a.hpp>

#include <cstddef>
#include <format>
#include <memory>
#include <vector>

namespace reticolo::llr {

// Shared LLR driver: stamps /cfg metadata, opens the per-replica /a /dE
// series + /exchange/accepted, runs the Newton-Raphson warm-up phase, then
// the Robbins-Monro phase with even/odd nearest-neighbour exchange. The four
// LLR apps differ only in (action type, field type, optional pre-NR warmup,
// optional extra HDF5 attrs); everything from "stamp cfg" through "drain RM
// buffers" is identical and lives here.
//
// Callers are expected to: (1) construct the replica vector and bind it to
// the writer's "llr" phase, (2) stamp any app-specific extra attrs on the
// writer before invoking `run`, (3) run any pre-NR work (e.g. the Bose gas
// Metropolis warmup cascade), (4) call `run`. The driver does not start
// the writer phase — that's the app's call so app-specific attrs can land
// at the right path.

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
    bool exchange = true;  // even/odd replica swaps each RM sweep
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

    auto e_n_series = out.series<double>("/cfg/E_n");
    for (auto const& r : reps) {
        e_n_series.append(r->E_n());
    }

    std::vector<io::Series<double>> a_series;
    std::vector<io::Series<double>> de_series;
    a_series.reserve(n_rep_u);
    de_series.reserve(n_rep_u);
    for (int n = 0; n < n_rep; ++n) {
        a_series.emplace_back(out.series<double>(std::format("/replica_{:03d}/a", n)));
        de_series.emplace_back(out.series<double>(std::format("/replica_{:03d}/dE", n)));
    }
    auto exch_series = out.series<int>("/exchange/accepted");

    // Per-iteration scratch — staged in parallel, drained serially because
    // HDF5 writes are not thread-safe.
    std::vector<double> de_buf(n_rep_u);
    std::vector<double> a_buf(n_rep_u);

    log::info("llr", "NR phase  {} iters × {} replicas", spec.n_nr, n_rep_u);
    for (int k = 0; k < spec.n_nr; ++k) {
#pragma omp parallel for schedule(dynamic, 1)
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            auto _  = log::scope(reps[n]->id());
            auto& r = *reps[n];
            r.thermalize(spec.n_therm_nr);
            de_buf[n] = r.sample(spec.n_meas_nr);
            a_buf[n]  = nr_update(r.a(), de_buf[n], spec.delta);
            r.set_a(a_buf[n]);
        }
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            a_series[n].append(a_buf[n]);
            de_series[n].append(de_buf[n]);
        }
        log::info("llr", "NR iter  {:>3}/{}  done", k + 1, spec.n_nr);
    }

    log::info("llr", "RM phase  {} iters × {} replicas", spec.n_rm, n_rep_u);
    for (int s = 0; s < spec.n_rm; ++s) {
#pragma omp parallel for schedule(dynamic, 1)
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            auto _  = log::scope(reps[n]->id());
            auto& r = *reps[n];
            r.thermalize(spec.n_therm_rm, log::Mode::silent);
            de_buf[n] = r.sample(spec.n_meas_rm, log::Mode::silent);
            a_buf[n]  = rm_update(r.a(), de_buf[n], spec.delta, s);
            r.set_a(a_buf[n]);
            iter("RM",
                 static_cast<std::size_t>(s + 1),
                 static_cast<std::size_t>(spec.n_rm),
                 a_buf[n],
                 de_buf[n],
                 spec.delta);
        }
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            a_series[n].append(a_buf[n]);
            de_series[n].append(de_buf[n]);
        }

        // Even/odd alternating nearest-neighbour exchange: serial — pairs
        // of replicas and a single shared exchange RNG. When disabled the
        // series still gets a 0 so its shape stays in lockstep with /a, /dE.
        int accepted = 0;
        int attempts = 0;
        if (spec.exchange) {
            std::size_t const off = static_cast<std::size_t>(s & 1);
            for (std::size_t i = off; i + 1 < reps.size(); i += 2) {
                ++attempts;
                if (try_exchange(*reps[i], *reps[i + 1], exch_rng)) {
                    ++accepted;
                }
            }
        }
        exch_series.append(accepted);
        log::info("exch", "step  {:>3}  accepted  {}/{}", s + 1, accepted, attempts);
        log::info("llr", "RM iter  {:>3}/{}  done", s + 1, spec.n_rm);
    }
}

}  // namespace reticolo::llr
