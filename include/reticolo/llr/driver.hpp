#pragma once

#include <reticolo/io/writer.hpp>
#include <reticolo/llr/checkpoint.hpp>
#include <reticolo/llr/exchange.hpp>
#include <reticolo/llr/log.hpp>
#include <reticolo/llr/update_a.hpp>

#include <cstddef>
#include <format>
#include <memory>
#include <string>
#include <vector>

#ifdef _OPENMP
    #include <omp.h>
#endif

namespace reticolo::llr {

// Shared LLR driver: stamps /cfg metadata, opens the per-replica /a /dE
// series + /exchange/accepted, then runs the full schedule — a windowed-HMC
// warm-up phase (drive each replica into its E_n window), the Newton-Raphson
// warm-up, then the Robbins-Monro phase with even/odd nearest-neighbour
// exchange. The LLR apps differ only in (action type, field type, optional
// hot start, optional extra HDF5 attrs); everything from "stamp cfg" through
// "drain RM buffers" is identical and lives here.
//
// Callers are expected to: (1) construct the replica vector and bind it to the
// writer's "llr" phase, (2) stamp any app-specific extra attrs on the writer
// before invoking `run`, (3) optionally hot-start the fields (field-specific —
// unsafe for matrix gauge groups, so gauge apps warm in from cold identity),
// (4) call `run`. The driver does not start the writer phase — that's the app's
// call so app-specific attrs can land at the right path.

struct DriverSpec {
    int n_nr{};
    int n_therm_nr{};
    int n_meas_nr{};
    int n_rm{};
    int n_therm_rm{};
    int n_meas_rm{};
    double delta{};
    double e_min{};
    // NOLINTNEXTLINE(readability-identifier-naming) physics convention
    double E_max{};  // already snapped to the (n_rep - 1) * delta grid
    double d_e{};
    bool exchange = true;  // even/odd replica swaps each RM sweep

    // ---- warm-up (seat each replica in its E_n window before NR) ----
    // Two-stage pre-phase run once on a fresh start (skipped on resume — the
    // checkpoint already holds warmed fields + tilt). Stage 1: warm_therm
    // trajectories of plain BASE-action HMC to reach statistical equilibrium.
    // Stage 2: coarse Newton a-adaptation until ⟨S⟩ sits within warm_thresh·δ of
    // E_n; the warmed `a` is carried into NR. warm_max_traj caps stage 2 (a
    // warning if hit). warm_therm = 0 skips stage 1; warm_max_traj = 0 disables
    // the whole phase.
    int warm_therm     = 200;
    int warm_max_traj  = 2000;
    double warm_thresh = 1.0;

    // ---- threading (see llr::plan_threads) ----
    // Per-replica HMC thread count `m` and slab granularity, recorded in the
    // checkpoint so a resume reconstructs the same partition. `concurrency` is
    // the outer replica-team size (num_threads on the replica loop); 0 = ambient
    // = today's flat model. `nested` (set when m>1) enables OpenMP's second
    // active level so a replica's HMC traverse can spawn its inner team.
    int replica_threads = 1;
    int slabs           = 0;
    int concurrency     = 0;
    bool nested         = false;

    // ---- checkpoint / resume ----
    // `checkpoint_path` empty ⇒ no checkpointing. `checkpoint_every` sweeps
    // between rolling snapshots (0 ⇒ only the final one). start_phase/start_iter
    // are set by the app from load_ensemble: (0,k) resumes NR at iter k, (1,s)
    // skips NR and resumes RM at sweep s. Default (0,0) runs the whole schedule.
    // Default member init so a partial DriverSpec{...} (the examples) can omit it
    // without tripping -Wmissing-designated-field-initializers.
    std::string checkpoint_path{};  // NOLINT(readability-redundant-member-init)
    int checkpoint_every = 0;
    int start_phase      = 0;
    int start_iter       = 0;
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

    // Small flush chunk for these low-cardinality per-iteration series: with the
    // default 4096 nothing reaches the file until the Series dtor at run end, so
    // a crash loses the whole adaptation history. 64 rows bounds the loss to a
    // handful of iterations at negligible HDF5 overhead.
    constexpr std::size_t k_adapt_chunk = 64;
    std::vector<io::Series<double>> a_series;
    std::vector<io::Series<double>> de_series;
    a_series.reserve(n_rep_u);
    de_series.reserve(n_rep_u);
    for (int n = 0; n < n_rep; ++n) {
        a_series.emplace_back(
            out.series<double>(std::format("/replica_{:03d}/a", n), k_adapt_chunk));
        de_series.emplace_back(
            out.series<double>(std::format("/replica_{:03d}/dE", n), k_adapt_chunk));
    }
    auto exch_series = out.series<int>("/exchange/accepted");

    // Per-iteration scratch — staged in parallel, drained serially because
    // HDF5 writes are not thread-safe.
    std::vector<double> de_buf(n_rep_u);
    std::vector<double> a_buf(n_rep_u);

    // Outer replica-team size + nested-region enable. num_threads(outer_nt)
    // pins the replica loop to `concurrency` lanes so `concurrency × m` inner
    // HMC threads don't oversubscribe; at m=1 (concurrency=0) outer_nt is the
    // ambient count and the clause is a no-op — today's flat behaviour.
#ifdef _OPENMP
    if (spec.nested) {
        omp_set_max_active_levels(2);
    }
    int const outer_nt = spec.concurrency > 0 ? spec.concurrency : omp_get_max_threads();
#else
    [[maybe_unused]] int const outer_nt = spec.concurrency > 0 ? spec.concurrency : 1;
#endif

    // One-shot run summary: ensemble geometry, the shared HMC sampler (announced
    // once via a representative replica — all share the same integrator / τ /
    // n_md), and the schedule + thread split. Per-replica construction is logged
    // under log::quiet() by the app, so this stands in for that boilerplate.
    log::info("llr",
              "ensemble  {} replicas · E_n ∈ [{:+.1f} … {:+.1f}] · dE={:.1f} · δ={:.1f}",
              n_rep,
              spec.e_min,
              spec.E_max,
              spec.d_e,
              spec.delta);
    if (!reps.empty()) {
        reps.front()->announce_sampler();
    }
    log::info("llr",
              "schedule  NR {}×(therm {}, meas {}) · RM {}×(therm {}, meas {}){}",
              spec.n_nr,
              spec.n_therm_nr,
              spec.n_meas_nr,
              spec.n_rm,
              spec.n_therm_rm,
              spec.n_meas_rm,
              spec.exchange ? " · exchange" : "");
    log::info("llr", "threads   m={} × {} concurrent", spec.replica_threads, outer_nt);

    // Warm-up phase: seat every replica in its E_n window by coarse tilt
    // adaptation (group-safe — plain HMC steps + Newton `a` updates, unlike a hot
    // start). Fresh runs only; a resume enters with already-warmed fields + tilt.
    if (spec.start_phase == 0 && spec.start_iter == 0 && spec.warm_max_traj > 0) {
        log::info("llr", "warm phase  seat {} replicas in window (a-adapting)", n_rep_u);
#pragma omp parallel for schedule(dynamic, 1) num_threads(outer_nt)
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            reps[n]->warm_into_window(spec.warm_therm, spec.warm_max_traj, spec.warm_thresh);
        }
    }

    // Rolling atomic snapshot at (phase, next-iter). No-op when no path is set;
    // otherwise runs serially (HDF5 is not thread-safe) after the sweep drains.
    auto checkpoint = [&](int phase, int next_iter) {
        if (spec.checkpoint_path.empty()) {
            return;
        }
        save_ensemble(spec.checkpoint_path,
                      reps,
                      exch_rng,
                      OrchState{.phase     = phase,
                                .iter      = next_iter,
                                .n_threads = spec.replica_threads,
                                .slabs     = spec.slabs});
    };

    if (spec.start_phase == 0) {
        log::info("llr", "NR phase  {} iters × {} replicas", spec.n_nr, n_rep_u);
        for (int k = spec.start_iter; k < spec.n_nr; ++k) {
#pragma omp parallel for schedule(dynamic, 1) num_threads(outer_nt)
            for (std::size_t n = 0; n < n_rep_u; ++n) {
                auto& r = *reps[n];
                r.thermalize(spec.n_therm_nr, log::Mode::silent);
                de_buf[n] = r.sample(spec.n_meas_nr, log::Mode::silent);
                a_buf[n]  = nr_update(r.a(), de_buf[n], spec.delta);
                r.set_a(a_buf[n]);
            }
            // Drain + per-replica NR row serially (same schema as RM), binding
            // each replica's scope since we're outside the Replica's methods.
            for (std::size_t n = 0; n < n_rep_u; ++n) {
                auto _ = log::scope(reps[n]->id());
                a_series[n].append(a_buf[n]);
                de_series[n].append(de_buf[n]);
                iter("NR",
                     static_cast<std::size_t>(k) + 1,
                     static_cast<std::size_t>(spec.n_nr),
                     a_buf[n],
                     de_buf[n],
                     spec.delta);
            }
            log::info("llr", "NR iter  {:>3}/{}  done", k + 1, spec.n_nr);
            if (spec.checkpoint_every > 0 && (k + 1) % spec.checkpoint_every == 0) {
                checkpoint(0, k + 1);
            }
        }
        // NR complete → next resume enters RM at sweep 0.
        checkpoint(1, 0);
    }

    int const rm_start = spec.start_phase == 1 ? spec.start_iter : 0;
    log::info("llr", "RM phase  {} iters × {} replicas", spec.n_rm, n_rep_u);
    for (int s = rm_start; s < spec.n_rm; ++s) {
#pragma omp parallel for schedule(dynamic, 1) num_threads(outer_nt)
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            auto& r = *reps[n];
            r.thermalize(spec.n_therm_rm, log::Mode::silent);
            de_buf[n] = r.sample(spec.n_meas_rm, log::Mode::silent);
            a_buf[n]  = rm_update(r.a(), de_buf[n], spec.delta, s);
            r.set_a(a_buf[n]);
        }
        // Drain + per-replica progress log serially: the iter() call takes the
        // global sink mutex and flushes two log files, so it must stay out of
        // the parallel region (mirrors smoothed_driver). Bind each replica's
        // scope here since we're no longer inside the Replica's own methods.
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            auto _ = log::scope(reps[n]->id());
            a_series[n].append(a_buf[n]);
            de_series[n].append(de_buf[n]);
            iter("RM",
                 static_cast<std::size_t>(s) + 1,
                 static_cast<std::size_t>(spec.n_rm),
                 a_buf[n],
                 de_buf[n],
                 spec.delta);
        }

        // Even/odd alternating nearest-neighbour exchange: serial — pairs
        // of replicas and a single shared exchange RNG. When disabled the
        // series still gets a 0 so its shape stays in lockstep with /a, /dE.
        int accepted = 0;
        int attempts = 0;
        if (spec.exchange) {
            auto const off = static_cast<std::size_t>(s & 1);
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

        // Snapshot AFTER exchange, at a sweep boundary: the saved fields include
        // this sweep's swaps and the resume re-enters at sweep s+1.
        bool const last = (s + 1) == spec.n_rm;
        if (last || (spec.checkpoint_every > 0 && (s + 1) % spec.checkpoint_every == 0)) {
            checkpoint(1, s + 1);
        }
    }
}

}  // namespace reticolo::llr
