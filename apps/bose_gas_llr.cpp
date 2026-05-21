// LLR (Gaussian-penalty) for the 4D self-interacting relativistic lattice
// Bose gas at finite chemical potential. Phase-quenched HMC samples
// `S_R = base.s_full(phi)`; the LLR window constrains the imaginary action
// observable `S_I = base.s_imag(phi)` via the `HasImagPart` dispatch in
// llr::WindowedAction.
//
// Output schema (HDF5):
//  /cfg@n_rep, /cfg@delta, /cfg@E_min, /cfg@E_max, /cfg@dE, /cfg@mu
//  /cfg/E_n              — n_rep values (window centres in S_I)
//  /replica_NNN/a        — series, one append per NR iter + per RM sweep
//  /replica_NNN/dE       — series, paired with /a (<S_I − E_n>)
//  /exchange/accepted    — series, one int per RM sweep
//
// arxiv:1910.11026 reproduces the paper's `<e^{iφ}>_pq(μ)` curve by feeding
// the per-µ output of this binary through examples/06_bose_gas_llr/analyze.py.

#include <reticolo/reticolo.hpp>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <format>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Action   = act::BoseGas<double>;
    using ReplicaT = llr::Replica<Action, FastRng, alg::integ::Omelyan2>;

    cli::Parser p{"bose_gas_llr", "LLR for the 4D Bose gas at finite chemical potential"};
    auto const& L      = p.opt<int>("L,size", 4, "linear lattice extent");
    auto const& ndim   = p.opt<int>("ndim", 4, "spacetime dimensions (4 in paper)");
    auto const& mass   = p.opt<double>("mass", 1.0, "bare mass m");
    auto const& lambda = p.opt<double>("lambda", 1.0, "quartic coupling lambda");
    auto const& mu     = p.opt<double>("mu", 1.0, "chemical potential mu");
    auto const& e_min  = p.opt<double>("E_min", -10.0, "lower S_I window centre");
    auto const& e_max  = p.opt<double>("E_max", 10.0, "upper S_I window centre");
    auto const& delta =
        p.opt<double>("delta",
                      2.0,
                      "single LLR tuning knob: Gaussian half-width AND replica spacing in S_I. "
                      "n_rep is derived so adjacent window centres are exactly `delta` apart.");
    auto const& tau  = p.opt<double>("tau", 1.0, "HMC trajectory length");
    auto const& n_md = p.opt<int>("n_md", 10, "MD steps per trajectory");
    auto const& n_nr = p.opt<int>("n_nr", 6, "Newton-Raphson warm-up iterations");
    auto const& n_therm_nr =
        p.opt<int>("n_therm_nr", 200, "thermalisation trajectories per NR iter");
    auto const& n_meas_nr = p.opt<int>("n_meas_nr", 1000, "measurement trajectories per NR iter");
    auto const& n_rm      = p.opt<int>("n_rm", 20, "Robbins-Monro sweeps");
    auto const& n_therm_rm =
        p.opt<int>("n_therm_rm", 100, "thermalisation trajectories per RM sweep");
    auto const& n_meas_rm = p.opt<int>("n_meas_rm", 500, "measurement trajectories per RM sweep");
    auto const& seed      = p.opt<unsigned long long>("seed", 42ULL, "RNG seed");
    auto const& outpath = p.opt<std::string>("out", std::string{"bose_gas_llr.h5"}, "HDF5 output");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    log::start(outpath);

    Lattice<std::complex<double>>::SizeVec shape(static_cast<std::size_t>(ndim),
                                                 static_cast<std::size_t>(L));
    Action const base{.mass = mass, .lambda = lambda, .mu = mu};
    log::act(base);

    int const n_rep  = std::max(2, static_cast<int>(std::lround((e_max - e_min) / delta)) + 1);
    double const d_e = delta;
    double const e_max_snapped = e_min + (static_cast<double>(n_rep - 1) * d_e);

    std::vector<std::unique_ptr<ReplicaT>> reps;
    reps.reserve(static_cast<std::size_t>(n_rep));
    for (int n = 0; n < n_rep; ++n) {
        double const e_n = e_min + (static_cast<double>(n) * d_e);
        reps.push_back(std::make_unique<ReplicaT>(
            base,
            FastRng{seed + 1ULL + static_cast<unsigned long long>(n)},
            ReplicaT::Spec{
                .id = std::format("r{:03}", n), .shape = shape, .e_n = e_n, .delta = delta},
            alg::HmcSpec{.tau = tau, .n_md = n_md}));
    }

    FastRng exch_rng{seed};
    io::Writer out{outpath, argc, argv, &p};
    out.start_phase("llr");

    out.attr<int>("/cfg@n_rep", n_rep);
    out.attr<int>("/cfg@n_nr", n_nr);
    out.attr<int>("/cfg@n_rm", n_rm);
    out.attr<double>("/cfg@delta", delta);
    out.attr<double>("/cfg@E_min", e_min);
    out.attr<double>("/cfg@E_max", e_max_snapped);
    out.attr<double>("/cfg@dE", d_e);
    out.attr<double>("/cfg@mu", mu);

    auto e_n_series = out.series<double>("/cfg/E_n");
    for (auto const& r : reps) {
        e_n_series.append(r->E_n());
    }

    std::vector<io::Series<double>> a_series;
    std::vector<io::Series<double>> de_series;
    a_series.reserve(static_cast<std::size_t>(n_rep));
    de_series.reserve(static_cast<std::size_t>(n_rep));
    for (int n = 0; n < n_rep; ++n) {
        a_series.emplace_back(out.series<double>(std::format("/replica_{:03d}/a", n)));
        de_series.emplace_back(out.series<double>(std::format("/replica_{:03d}/dE", n)));
    }
    auto exch_series = out.series<int>("/exchange/accepted");

    std::size_t const n_rep_u = static_cast<std::size_t>(n_rep);

    // Replicas are independent (each owns its own field / RNG / base action
    // copy via WindowedAction-by-value), so the per-replica loops below are
    // clean OpenMP parallel-for. HDF5 writes are not thread-safe — stage
    // into per-iteration buffers, drain serially.
    std::vector<double> de_buf(n_rep_u);
    std::vector<double> a_buf(n_rep_u);

    // Hot-start every replica with an independent random field, then run
    // windowed Metropolis until each replica is inside its E_n window.
    // HMC alone can't reliably pull a replica into the window at the S_I
    // tail: the windowed force gets large enough that the integrator goes
    // unstable, dH explodes, accept collapses. Metropolis with a Gaussian-
    // shift proposal stays stable at any window strength; the LLR tilt is
    // wired into WindowedAction's ds_local via begin_sweep / commit_accept.
    constexpr double k_hot_sigma          = 0.5;
    constexpr double k_metro_sigma        = 0.3;
    constexpr int k_metro_max_batches     = 50;
    constexpr int k_metro_batch_size      = 10;
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t n = 0; n < n_rep_u; ++n) {
        auto _ = log::scope(reps[n]->id());
        reps[n]->hot_start(k_hot_sigma);
        alg::Metropolis<llr::WindowedAction<Action>, FastRng> warmup{
            reps[n]->windowed_action(),
            reps[n]->phi(),
            reps[n]->rng(),
            alg::MetropolisSpec{.sigma = k_metro_sigma},
            log::Mode::silent};
        reps[n]->thermalize_until_in_window(
            warmup, k_metro_max_batches, k_metro_batch_size, 1.0);
    }

    log::info("llr", "NR phase  {} iters × {} replicas", n_nr, n_rep_u);
    for (int k = 0; k < n_nr; ++k) {
#pragma omp parallel for schedule(dynamic, 1)
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            auto _  = log::scope(reps[n]->id());
            auto& r = *reps[n];
            r.thermalize(n_therm_nr);
            de_buf[n] = r.sample(n_meas_nr);
            a_buf[n]  = llr::nr_update(r.a(), de_buf[n], delta);
            r.set_a(a_buf[n]);
        }
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            a_series[n].append(a_buf[n]);
            de_series[n].append(de_buf[n]);
        }
        log::info("llr", "NR iter  {:>3}/{}  done", k + 1, n_nr);
    }

    log::info("llr", "RM phase  {} iters × {} replicas", n_rm, n_rep_u);
    for (int s = 0; s < n_rm; ++s) {
#pragma omp parallel for schedule(dynamic, 1)
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            auto _  = log::scope(reps[n]->id());
            auto& r = *reps[n];
            r.thermalize(n_therm_rm, log::Mode::silent);
            de_buf[n] = r.sample(n_meas_rm, log::Mode::silent);
            a_buf[n]  = llr::rm_update(r.a(), de_buf[n], delta, s);
            r.set_a(a_buf[n]);
            llr::iter("RM",
                      static_cast<std::size_t>(s + 1),
                      static_cast<std::size_t>(n_rm),
                      a_buf[n],
                      de_buf[n],
                      delta);
        }
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            a_series[n].append(a_buf[n]);
            de_series[n].append(de_buf[n]);
        }

        // Even/odd alternating nearest-neighbour exchange — serial: touches
        // pairs of replicas and a single shared exchange RNG.
        std::size_t const off = static_cast<std::size_t>(s & 1);
        int accepted          = 0;
        int attempts          = 0;
        for (std::size_t i = off; i + 1 < reps.size(); i += 2) {
            ++attempts;
            if (llr::try_exchange(*reps[i], *reps[i + 1], exch_rng)) {
                ++accepted;
            }
        }
        exch_series.append(accepted);
        log::info("exch", "step  {:>3}  accepted  {}/{}", s + 1, accepted, attempts);
        log::info("llr", "RM iter  {:>3}/{}  done", s + 1, n_rm);
    }
}
