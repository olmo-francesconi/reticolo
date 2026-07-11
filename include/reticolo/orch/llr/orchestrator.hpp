#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/io/writer.hpp>
#include <reticolo/orch/checkpoint.hpp>
#include <reticolo/orch/ensemble.hpp>
#include <reticolo/orch/llr/checkpoint.hpp>
#include <reticolo/orch/llr/exchange.hpp>
#include <reticolo/orch/llr/log.hpp>
#include <reticolo/orch/llr/replica.hpp>
#include <reticolo/orch/llr/update_a.hpp>
#include <reticolo/orch/thread_plan.hpp>
#include <reticolo/updater/hmc/integrators.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <memory>
#include <string>
#include <vector>

namespace reticolo::orch::llr {

// LLR orchestrator: the whole replica-exchange LLR run as one object with a
// two-phase lifecycle — setup() wires everything up (thread plan, the replica
// ladder, resume, IO), run() / run_smoothed() do the computation. It replaces
// the free `orch::llr::run` / `smoothed::run` drivers; the schedule (windowed-HMC
// warm-up → Newton-Raphson warm-up → Robbins-Monro with even/odd exchange) is
// identical, and every /cfg, /replica_NNN and /exchange dataset is byte-for-byte
// the same, so old checkpoints resume unchanged.
//
// The disputed per-replica construction is pure boilerplate for LLR — the window
// centres are a uniform ladder E_n = E_min + n·dE and the seeds derive from one
// base seed — so setup() owns it. The ONE thing that genuinely varies per app is
// the initial field content (real scalars start cold; gauge SU(N) cold-starts to
// identity; U(1) / complex hot-start): the app does that between setup() and run()
// via replicas(), guarded by resuming().
//
//   using Llr = orch::llr::Orchestrator<act::Phi4<double>, FastRng>;
//   Llr llr{base, orch::llr::Spec{.shape=shape, .seed=seed, .e_min=…, …}};
//   io::Writer out{outpath, argc, argv, &p};
//   out.start_phase("llr");
//   llr.setup(out);
//   llr.run();

// Everything the run needs beyond the base action + window constraint: window
// geometry (in the constraint observable's units), the shared HMC sampler, the
// NR/RM schedule, warm-up, threading and checkpoint/resume. `spacing = 0` makes
// the replica spacing equal to `delta`; the ladder centres are snapped to a
// (n_rep − 1)·dE grid, exactly as the old apps computed inline.
struct Spec {
    std::vector<std::size_t> shape;
    unsigned long long seed = 0;

    double e_min   = 0.0;
    double e_max   = 0.0;
    double delta   = 0.0;  // Gaussian penalty width δ AND (if spacing=0) replica spacing
    double spacing = 0.0;  // window-centre interval; 0 ⇒ equal to delta

    double tau = 1.0;
    int n_md   = 20;

    int n_nr       = 6;
    int n_therm_nr = 200;
    int n_meas_nr  = 1000;
    int n_rm       = 20;
    int n_therm_rm = 100;
    int n_meas_rm  = 500;
    bool exchange  = true;

    int warm_therm     = 200;
    int warm_max_traj  = 2000;
    double warm_thresh = 1.0;

    int replica_threads = 1;  // → plan_threads(n_rep, replica_threads)
    int slabs           = 0;

    std::string checkpoint_path;  // NOLINT(readability-redundant-member-init)
    std::string resume;           // NOLINT(readability-redundant-member-init)
    int checkpoint_every = 0;
};

// Smoothed-RM knobs: a cross-replica local-polynomial fit â(E) shrunk into each
// per-replica RM iterate. λ_s = λ0/(s+1)^exp; exp > 1 keeps the perturbation
// summable so the fixed point is unchanged from vanilla LLR.
struct SmoothConfig {
    int smooth_K             = 4;  // NOLINT(readability-identifier-naming) physics convention
    int smooth_degree        = 2;
    double smooth_lambda0    = 1.0;
    double smooth_lambda_exp = 2.0;
};

namespace detail {

// In-place Gauss elimination with partial pivoting. `mat` is `n × n` row-major;
// `vec` is `n`. Returns false if singular to within `k_tol`; else `vec` holds
// the solution.
inline bool gauss_solve(std::vector<double>& mat, std::vector<double>& vec, int n) {
    constexpr double k_tol = 1e-12;
    auto const ni          = static_cast<std::size_t>(n);
    auto idx               = [ni](int i, int j) {
        return (static_cast<std::size_t>(i) * ni) + static_cast<std::size_t>(j);
    };
    for (int k = 0; k < n; ++k) {
        int pivot          = k;
        double pivot_value = std::abs(mat[idx(k, k)]);
        for (int i = k + 1; i < n; ++i) {
            double const v = std::abs(mat[idx(i, k)]);
            if (v > pivot_value) {
                pivot_value = v;
                pivot       = i;
            }
        }
        if (pivot_value < k_tol) {
            return false;
        }
        if (pivot != k) {
            for (int j = 0; j < n; ++j) {
                std::swap(mat[idx(k, j)], mat[idx(pivot, j)]);
            }
            std::swap(vec[static_cast<std::size_t>(k)], vec[static_cast<std::size_t>(pivot)]);
        }
        double const inv_pk = 1.0 / mat[idx(k, k)];
        for (int i = k + 1; i < n; ++i) {
            double const factor = mat[idx(i, k)] * inv_pk;
            for (int j = k; j < n; ++j) {
                mat[idx(i, j)] -= factor * mat[idx(k, j)];
            }
            vec[static_cast<std::size_t>(i)] -= factor * vec[static_cast<std::size_t>(k)];
        }
    }
    for (int i = n - 1; i >= 0; --i) {
        double sum = vec[static_cast<std::size_t>(i)];
        for (int j = i + 1; j < n; ++j) {
            sum -= mat[idx(i, j)] * vec[static_cast<std::size_t>(j)];
        }
        vec[static_cast<std::size_t>(i)] = sum / mat[idx(i, i)];
    }
    return true;
}

// Local-polynomial fit of degree `deg` through the (2K+1) replicas centred on
// index `n` (clamped at the ends), evaluated at E = e[n]. Falls back to the raw
// value `a[n]` when fewer than `deg + 1` points are available.
// NOLINTNEXTLINE(readability-identifier-naming, bugprone-easily-swappable-parameters)
[[nodiscard]] inline double local_poly_fit_at(std::vector<double> const& e,
                                              std::vector<double> const& a,
                                              std::size_t n,
                                              int K,  // NOLINT
                                              int deg) {
    int const total = static_cast<int>(e.size());
    int const ni    = static_cast<int>(n);
    int const lo    = std::max(0, ni - K);
    int const hi    = std::min(total - 1, ni + K);
    int const npts  = (hi - lo) + 1;
    int const dim   = deg + 1;
    if (npts < dim) {
        return a[n];
    }

    double const e0 = e[n];
    std::vector<double> moms(static_cast<std::size_t>((2 * deg) + 1), 0.0);
    std::vector<double> rhs(static_cast<std::size_t>(dim), 0.0);
    for (int m = lo; m <= hi; ++m) {
        double const x = e[static_cast<std::size_t>(m)] - e0;
        double const y = a[static_cast<std::size_t>(m)];
        double xp      = 1.0;
        for (int p = 0; p <= 2 * deg; ++p) {
            moms[static_cast<std::size_t>(p)] += xp;
            if (p <= deg) {
                rhs[static_cast<std::size_t>(p)] += xp * y;
            }
            xp *= x;
        }
    }
    auto const dim_z = static_cast<std::size_t>(dim);
    std::vector<double> mat(dim_z * dim_z, 0.0);
    for (int i = 0; i < dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            mat[(static_cast<std::size_t>(i) * dim_z) + static_cast<std::size_t>(j)] =
                moms[static_cast<std::size_t>(i) + static_cast<std::size_t>(j)];
        }
    }
    if (!gauss_solve(mat, rhs, dim)) {
        return a[n];
    }
    return rhs[0];
}

inline void local_poly_fit(std::vector<double> const& e,
                           std::vector<double> const& a,
                           int K,  // NOLINT(readability-identifier-naming)
                           int deg,
                           std::vector<double>& out) {
    std::size_t const total = e.size();
    out.assign(total, 0.0);
    for (std::size_t n = 0; n < total; ++n) {
        out[n] = local_poly_fit_at(e, a, n, K, deg);
    }
}

}  // namespace detail

template <class Base,
          class Rng,
          class Integrator = updater::integ::Omelyan2,
          class T          = Base::value_type,
          class Field      = Lattice<T>,
          class Constraint = void>
class Orchestrator {
public:
    using replica_type    = Replica<Base, Rng, Integrator, T, Field, Constraint>;
    using constraint_type = replica_type::constraint_type;
    using scalar_t        = action::scalar_of_t<T>;

    Orchestrator(Base base, Spec spec, constraint_type constraint = {})
        : base_{std::move(base)}, spec_{std::move(spec)}, constraint_{std::move(constraint)},
          exch_rng_{spec_.seed} {}

    // Plan threads, build the replica ladder, resume if requested, stamp the
    // standard /cfg metadata and open the per-replica /a /dE + /exchange series.
    // Announces the ensemble; the per-replica ctors are logged under log::quiet.
    // The writer's phase must already be started by the app (so app-specific
    // attrs land at the right path).
    void setup(io::Writer& out) {
        out_ = &out;

        d_e_   = spec_.spacing > 0.0 ? spec_.spacing : spec_.delta;
        n_rep_ = std::max(2, static_cast<int>(std::lround((spec_.e_max - spec_.e_min) / d_e_)) + 1);
        e_max_snapped_     = spec_.e_min + (static_cast<double>(n_rep_ - 1) * d_e_);
        plan_              = orch::plan_threads(n_rep_, spec_.replica_threads);
        auto const n_rep_u = static_cast<std::size_t>(n_rep_);

        reps_.reserve(n_rep_u);
        {
            auto const quiet = log::quiet();  // silence per-replica ctor announces
            for (int n = 0; n < n_rep_; ++n) {
                double const e_n = spec_.e_min + (static_cast<double>(n) * d_e_);
                reps_.push_back(std::make_unique<replica_type>(
                    base_,
                    Rng{spec_.seed + 1ULL + static_cast<unsigned long long>(n)},
                    typename replica_type::Spec{.id    = std::format("r{:03}", n),
                                                .shape = spec_.shape,
                                                .e_n   = static_cast<scalar_t>(e_n),
                                                .delta = static_cast<scalar_t>(spec_.delta)},
                    updater::HmcSpec{.tau              = spec_.tau,
                                     .n_md             = spec_.n_md,
                                     .n_threads        = plan_.m,
                                     .slabs_per_thread = spec_.slabs},
                    constraint_));
            }
        }

        // Resume: restore fields / RNG streams / a / exch_rng before driving.
        if (!spec_.resume.empty()) {
            resume_state_ = load_ensemble(spec_.resume, reps_, exch_rng_);
            resuming_     = true;
            log::info("llr",
                      "resumed from {}  phase={} iter={}",
                      spec_.resume,
                      resume_state_.phase,
                      resume_state_.iter);
        }

        out.attr<int>("/cfg@n_rep", n_rep_);
        out.attr<int>("/cfg@n_nr", spec_.n_nr);
        out.attr<int>("/cfg@n_rm", spec_.n_rm);
        out.attr<double>("/cfg@delta", spec_.delta);
        out.attr<double>("/cfg@E_min", spec_.e_min);
        out.attr<double>("/cfg@E_max", e_max_snapped_);
        out.attr<double>("/cfg@dE", d_e_);

        auto e_n_series = out.series<double>("/cfg/E_n");
        e_n_vec_.reserve(n_rep_u);
        for (auto const& r : reps_) {
            auto const en = r->E_n();
            e_n_series.append(en);
            e_n_vec_.push_back(static_cast<double>(en));
        }

        // Small flush chunk for these low-cardinality per-iteration series so a
        // crash loses only a handful of iterations (see the old driver).
        constexpr std::size_t k_adapt_chunk = 64;
        a_series_.reserve(n_rep_u);
        de_series_.reserve(n_rep_u);
        for (int n = 0; n < n_rep_; ++n) {
            a_series_.emplace_back(
                out.series<double>(std::format("/replica_{:03d}/a", n), k_adapt_chunk));
            de_series_.emplace_back(
                out.series<double>(std::format("/replica_{:03d}/dE", n), k_adapt_chunk));
        }
        exch_series_ = out.series<int>("/exchange/accepted");

        de_buf_.assign(n_rep_u, 0.0);
        a_buf_.assign(n_rep_u, 0.0);

        log::info("llr",
                  "ensemble  {} replicas · E_n ∈ [{:+.1f} … {:+.1f}] · dE={:.1f} · δ={:.1f}",
                  n_rep_,
                  spec_.e_min,
                  e_max_snapped_,
                  d_e_,
                  spec_.delta);
        if (!reps_.empty()) {
            reps_.front()->announce_sampler();
        }
        log::info("llr", "threads   m={} × {} concurrent", plan_.m, plan_.concurrency);
    }

    // Vanilla LLR: NR warm-up + Robbins-Monro with even/odd exchange.
    void run() {
        log::info("llr",
                  "schedule  NR {}×(therm {}, meas {}) · RM {}×(therm {}, meas {}){}",
                  spec_.n_nr,
                  spec_.n_therm_nr,
                  spec_.n_meas_nr,
                  spec_.n_rm,
                  spec_.n_therm_rm,
                  spec_.n_meas_rm,
                  spec_.exchange ? " · exchange" : "");
        warm_fresh();
        if (start_phase() == 0) {
            nr_phase();
        }
        rm_phase_vanilla();
    }

    // Smoothed LLR: NR warm-up + smoothed Robbins-Monro (local-polynomial
    // shrinkage) with even/odd exchange. Opens four extra per-replica series
    // (/a_pre_shrink, /a_hat, /da_rm, /da_sm) and stamps the smoother /cfg attrs.
    void run_smoothed(SmoothConfig const& sm) {
        auto const n_rep_u = reps_.size();

        out_->attr<int>("/cfg@smooth_K", sm.smooth_K);
        out_->attr<int>("/cfg@smooth_degree", sm.smooth_degree);
        out_->attr<double>("/cfg@smooth_lambda0", sm.smooth_lambda0);
        out_->attr<double>("/cfg@smooth_lambda_exp", sm.smooth_lambda_exp);

        a_pre_series_.reserve(n_rep_u);
        a_hat_series_.reserve(n_rep_u);
        drm_series_.reserve(n_rep_u);
        dsm_series_.reserve(n_rep_u);
        for (int n = 0; n < n_rep_; ++n) {
            a_pre_series_.emplace_back(
                out_->series<double>(std::format("/replica_{:03d}/a_pre_shrink", n)));
            a_hat_series_.emplace_back(
                out_->series<double>(std::format("/replica_{:03d}/a_hat", n)));
            drm_series_.emplace_back(out_->series<double>(std::format("/replica_{:03d}/da_rm", n)));
            dsm_series_.emplace_back(out_->series<double>(std::format("/replica_{:03d}/da_sm", n)));
        }

        log::info("llr",
                  "schedule  NR {}×(therm {}, meas {}) · sRM {}×(therm {}, meas {}){} · "
                  "smooth K={} deg={} λ0={:.2g} exp={:.2g}",
                  spec_.n_nr,
                  spec_.n_therm_nr,
                  spec_.n_meas_nr,
                  spec_.n_rm,
                  spec_.n_therm_rm,
                  spec_.n_meas_rm,
                  spec_.exchange ? " · exchange" : "",
                  sm.smooth_K,
                  sm.smooth_degree,
                  sm.smooth_lambda0,
                  sm.smooth_lambda_exp);
        warm_fresh();
        if (start_phase() == 0) {
            nr_phase();
        }
        rm_phase_smoothed(sm);
    }

    // App-side field initialisation (cold-to-identity / hot-start) happens here,
    // between setup() and run(), guarded by resuming().
    [[nodiscard]] std::vector<std::unique_ptr<replica_type>>& replicas() noexcept { return reps_; }
    [[nodiscard]] bool resuming() const noexcept { return resuming_; }
    [[nodiscard]] int n_rep() const noexcept { return n_rep_; }

private:
    [[nodiscard]] int start_phase() const noexcept { return resuming_ ? resume_state_.phase : 0; }
    [[nodiscard]] int start_iter() const noexcept { return resuming_ ? resume_state_.iter : 0; }

    // Seat every replica in its E_n window (coarse Newton a-adaptation — group
    // safe, unlike a hot start). Fresh runs only; a resume enters warmed.
    void warm_fresh() {
        if (start_phase() != 0 || start_iter() != 0 || spec_.warm_max_traj <= 0) {
            return;
        }
        log::info("llr", "warm phase  seat {} replicas in window (a-adapting)", reps_.size());
        orch::parallel_workers(reps_, plan_, [&](std::size_t /*n*/, replica_type& r) {
            r.warm_into_window(spec_.warm_therm, spec_.warm_max_traj, spec_.warm_thresh);
        });
    }

    void checkpoint(int phase, int next_iter) {
        if (spec_.checkpoint_path.empty()) {
            return;
        }
        save_ensemble(
            spec_.checkpoint_path,
            reps_,
            exch_rng_,
            OrchState{
                .phase = phase, .iter = next_iter, .n_threads = plan_.m, .slabs = spec_.slabs});
    }

    // Even/odd alternating nearest-neighbour exchange. Serial (shared exch RNG).
    // Appends to /exchange/accepted (a 0 when disabled keeps the series shape in
    // lockstep with /a, /dE) and logs the step.
    void do_exchange(int s) {
        int accepted = 0;
        int attempts = 0;
        if (spec_.exchange) {
            auto const off = static_cast<std::size_t>(s & 1);
            for (std::size_t i = off; i + 1 < reps_.size(); i += 2) {
                ++attempts;
                if (try_exchange(*reps_[i], *reps_[i + 1], exch_rng_)) {
                    ++accepted;
                }
            }
        }
        exch_series_.append(accepted);
        log::info("exch", "step  {:>3}  accepted  {}/{}", s + 1, accepted, attempts);
    }

    // NR warm-up (shared by run / run_smoothed). Writes the smoothed extra series
    // with NR placeholders when they are open. Ends with checkpoint(1, 0).
    void nr_phase() {
        auto const n_rep_u = reps_.size();
        bool const smooth  = !a_pre_series_.empty();
        log::info("llr", "NR phase  {} iters × {} replicas", spec_.n_nr, n_rep_u);
        for (int k = start_iter(); k < spec_.n_nr; ++k) {
            orch::parallel_workers(reps_, plan_, [&](std::size_t n, replica_type& r) {
                r.thermalize(spec_.n_therm_nr, log::Mode::silent);
                de_buf_[n] = static_cast<double>(r.sample(spec_.n_meas_nr, log::Mode::silent));
            });
            for (std::size_t n = 0; n < n_rep_u; ++n) {
                auto& r   = *reps_[n];
                a_buf_[n] = nr_update(r.a(), de_buf_[n], spec_.delta);
                r.set_a(static_cast<scalar_t>(a_buf_[n]));
                auto _ = log::scope(r.id());
                a_series_[n].append(a_buf_[n]);
                de_series_[n].append(de_buf_[n]);
                if (smooth) {
                    a_pre_series_[n].append(a_buf_[n]);
                    a_hat_series_[n].append(a_buf_[n]);
                    drm_series_[n].append(0.0);
                    dsm_series_[n].append(0.0);
                }
                iter("NR",
                     static_cast<std::size_t>(k) + 1,
                     static_cast<std::size_t>(spec_.n_nr),
                     a_buf_[n],
                     de_buf_[n],
                     spec_.delta);
            }
            log::info("llr", "NR iter  {:>3}/{}  done", k + 1, spec_.n_nr);
            if (spec_.checkpoint_every > 0 && (k + 1) % spec_.checkpoint_every == 0) {
                checkpoint(0, k + 1);
            }
        }
        checkpoint(1, 0);
    }

    void rm_phase_vanilla() {
        auto const n_rep_u = reps_.size();
        int const rm_start = start_phase() == 1 ? start_iter() : 0;
        log::info("llr", "RM phase  {} iters × {} replicas", spec_.n_rm, n_rep_u);
        for (int s = rm_start; s < spec_.n_rm; ++s) {
            orch::parallel_workers(reps_, plan_, [&](std::size_t n, replica_type& r) {
                r.thermalize(spec_.n_therm_rm, log::Mode::silent);
                de_buf_[n] = static_cast<double>(r.sample(spec_.n_meas_rm, log::Mode::silent));
            });
            for (std::size_t n = 0; n < n_rep_u; ++n) {
                auto& r   = *reps_[n];
                a_buf_[n] = rm_update(r.a(), de_buf_[n], spec_.delta, s);
                r.set_a(static_cast<scalar_t>(a_buf_[n]));
                auto _ = log::scope(r.id());
                a_series_[n].append(a_buf_[n]);
                de_series_[n].append(de_buf_[n]);
                iter("RM",
                     static_cast<std::size_t>(s) + 1,
                     static_cast<std::size_t>(spec_.n_rm),
                     a_buf_[n],
                     de_buf_[n],
                     spec_.delta);
            }
            do_exchange(s);
            log::info("llr", "RM iter  {:>3}/{}  done", s + 1, spec_.n_rm);
            bool const last = (s + 1) == spec_.n_rm;
            if (last || (spec_.checkpoint_every > 0 && (s + 1) % spec_.checkpoint_every == 0)) {
                checkpoint(1, s + 1);
            }
        }
    }

    void rm_phase_smoothed(SmoothConfig const& sm) {
        auto const n_rep_u = reps_.size();
        std::vector<double> a_rm(n_rep_u);
        std::vector<double> a_hat(n_rep_u);
        std::vector<double> drm_buf(n_rep_u);
        std::vector<double> dsm_buf(n_rep_u);
        log::info(
            "llr",
            "smoothed RM phase  {} iters × {} replicas  (K={}, deg={}, λ0={:.2g}, exp={:.2g})",
            spec_.n_rm,
            n_rep_u,
            sm.smooth_K,
            sm.smooth_degree,
            sm.smooth_lambda0,
            sm.smooth_lambda_exp);
        int const rm_start = start_phase() == 1 ? start_iter() : 0;
        for (int s = rm_start; s < spec_.n_rm; ++s) {
            orch::parallel_workers(reps_, plan_, [&](std::size_t n, replica_type& r) {
                r.thermalize(spec_.n_therm_rm, log::Mode::silent);
                de_buf_[n] = static_cast<double>(r.sample(spec_.n_meas_rm, log::Mode::silent));
            });
            for (std::size_t n = 0; n < n_rep_u; ++n) {
                a_rm[n] = rm_update(reps_[n]->a(), de_buf_[n], spec_.delta, s);
            }

            detail::local_poly_fit(e_n_vec_, a_rm, sm.smooth_K, sm.smooth_degree, a_hat);
            double const lam =
                sm.smooth_lambda0 / std::pow(static_cast<double>(s + 1), sm.smooth_lambda_exp);
            double rm_step_sum = 0.0;
            double rm_step_max = 0.0;
            double sm_step_sum = 0.0;
            double sm_step_max = 0.0;
            for (std::size_t n = 0; n < n_rep_u; ++n) {
                auto const a_old = static_cast<double>(reps_[n]->a());
                drm_buf[n]       = std::abs(a_rm[n] - a_old);
                dsm_buf[n]       = std::abs(lam * (a_hat[n] - a_rm[n]));
                rm_step_sum += drm_buf[n];
                sm_step_sum += dsm_buf[n];
                rm_step_max = std::max(drm_buf[n], rm_step_max);
                sm_step_max = std::max(dsm_buf[n], sm_step_max);
                a_buf_[n]   = ((1.0 - lam) * a_rm[n]) + (lam * a_hat[n]);
                reps_[n]->set_a(static_cast<scalar_t>(a_buf_[n]));
                auto _ = log::scope(reps_[n]->id());
                iter("sRM",
                     static_cast<std::size_t>(s) + 1,
                     static_cast<std::size_t>(spec_.n_rm),
                     a_buf_[n],
                     de_buf_[n],
                     spec_.delta);
            }
            double const inv_n      = 1.0 / static_cast<double>(n_rep_u);
            double const rm_mean    = rm_step_sum * inv_n;
            double const sm_mean    = sm_step_sum * inv_n;
            double const ratio_mean = (rm_mean > 0.0) ? (sm_mean / rm_mean) : 0.0;
            double const ratio_max  = (rm_step_max > 0.0) ? (sm_step_max / rm_step_max) : 0.0;
            log::info("llr",
                      "sRM step  {:>3}/{}  λ={:.3e}  |Δa_rm| mean={:.2e} max={:.2e}  "
                      "|Δa_sm| mean={:.2e} max={:.2e}  sm/rm mean={:.2f} max={:.2f}",
                      s + 1,
                      spec_.n_rm,
                      lam,
                      rm_mean,
                      rm_step_max,
                      sm_mean,
                      sm_step_max,
                      ratio_mean,
                      ratio_max);

            for (std::size_t n = 0; n < n_rep_u; ++n) {
                a_series_[n].append(a_buf_[n]);
                de_series_[n].append(de_buf_[n]);
                a_pre_series_[n].append(a_rm[n]);
                a_hat_series_[n].append(a_hat[n]);
                drm_series_[n].append(drm_buf[n]);
                dsm_series_[n].append(dsm_buf[n]);
            }

            do_exchange(s);
            bool const last = (s + 1) == spec_.n_rm;
            if (last || (spec_.checkpoint_every > 0 && (s + 1) % spec_.checkpoint_every == 0)) {
                checkpoint(1, s + 1);
            }
        }
    }

    Base base_;
    Spec spec_;
    constraint_type constraint_;
    Rng exch_rng_;

    io::Writer* out_ = nullptr;
    ThreadPlan plan_{};
    int n_rep_            = 0;
    double d_e_           = 0.0;
    double e_max_snapped_ = 0.0;
    bool resuming_        = false;
    OrchState resume_state_{};

    std::vector<std::unique_ptr<replica_type>> reps_;
    std::vector<double> e_n_vec_;
    std::vector<io::Series<double>> a_series_;
    std::vector<io::Series<double>> de_series_;
    io::Series<int> exch_series_;
    std::vector<io::Series<double>> a_pre_series_;
    std::vector<io::Series<double>> a_hat_series_;
    std::vector<io::Series<double>> drm_series_;
    std::vector<io::Series<double>> dsm_series_;
    std::vector<double> de_buf_;
    std::vector<double> a_buf_;
};

}  // namespace reticolo::orch::llr
