// Phi4 Metropolis tuning rig. Per-update wall time + S + Σφ² to HDF5.
//
// Output schema:
//  /run@*               reproducibility metadata
//  /vars@*              every --flag the Parser resolved
//  /prod@wall_seconds   inner-loop wall time (double)
//  /prod@algo_seconds   step() wall time
//  /prod@obs_seconds    observable wall time
//  /prod@n_meas         number of production measurements
//  /prod@accept_rate    accepted/attempts over the production loop
//  /prod/obs/s          S_full per update
//  /prod/obs/mean_sq    Σφ² / V per update

#include <reticolo/reticolo.hpp>

#include "_tune/state_io.hpp"

#include <chrono>
#include <cstddef>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;
    using bench_clock = std::chrono::steady_clock;

    log::off();  // bench measurement: keep step-format cost out of the timing

    // ---- CLI ----
    cli::Parser p{"tune_phi4_metropolis", "Phi4 Metropolis tuning rig"};
    auto const& L         = p.opt<int>("L,size", 8, "linear lattice extent");
    auto const& kappa     = p.opt<double>("kappa", 0.18, "hopping parameter");
    auto const& lambda    = p.opt<double>("lambda", 1.0, "quartic coupling");
    auto const& ndim      = p.opt<int>("ndim", 3, "spatial dimensions");
    auto const& sigma     = p.opt<double>("sigma", 0.4, "Metropolis proposal width");
    auto const& n_therm   = p.opt<int>("n_therm", 1000, "thermalisation sweeps");
    auto const& n_prod    = p.opt<int>("n_prod", 5000, "production sweeps");
    auto const& seed      = p.opt<unsigned long long>("seed", 42ULL, "RNG seed");
    auto const& outpath   = p.opt<std::string>("out", std::string{"tune.h5"}, "HDF5 output");
    auto const& init_from = p.opt<std::string>(
        "init_from", std::string{}, "raw-binary state to load (skip thermalisation)");
    auto const& save_state = p.opt<std::string>(
        "save_state", std::string{}, "raw-binary state to save after production");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    // ---- State: lattice, RNG, action (+ optional resume) ----
    Lattice<double>::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(L));
    Lattice<double> phi{shape};
    FastRng rng{seed};
    act::Phi4<double> const phi4{.kappa = kappa, .lambda = lambda};

    if (!init_from.empty()) {
        tune::load_field_raw(phi, init_from);
    }

    // ---- Output ----
    io::Writer out{outpath, argc, argv, &p};
    out.start_phase("prod");

    // ---- Updater ----
    alg::Metropolis<act::Phi4<double>, FastRng> mc{
        phi4, phi, rng, alg::MetropolisSpec{.sigma = sigma}};

    // ---- Thermalisation ----
    for (int i = 0; i < n_therm; ++i) {
        (void)mc.step();
    }

    // ---- Production (timed) ----
    auto s_series       = out.series<double>("/prod/obs/s");
    auto mean_sq_series = out.series<double>("/prod/obs/mean_sq");

    std::size_t accepted = 0;
    std::size_t attempts = 0;
    double algo_s        = 0.0;
    double obs_s         = 0.0;
    auto const t_wall0   = bench_clock::now();
    for (int i = 0; i < n_prod; ++i) {
        auto const ta    = bench_clock::now();
        auto const stats = mc.step();
        auto const tb    = bench_clock::now();
        algo_s += std::chrono::duration<double>(tb - ta).count();
        accepted += stats.accepted;
        attempts += stats.attempts;

        auto const oa = bench_clock::now();
        s_series.append(phi4.s_full(phi));
        mean_sq_series.append(obs::sq_of_mean(phi));
        auto const ob = bench_clock::now();
        obs_s += std::chrono::duration<double>(ob - oa).count();
    }
    double const wall_s = std::chrono::duration<double>(bench_clock::now() - t_wall0).count();

    // ---- Stats ----
    out.attr<double>("/prod@wall_seconds", wall_s);
    out.attr<double>("/prod@algo_seconds", algo_s);
    out.attr<double>("/prod@obs_seconds", obs_s);
    out.attr<int>("/prod@n_meas", n_prod);
    out.attr<double>("/prod@accept_rate",
                     attempts == 0 ? 0.0
                                   : static_cast<double>(accepted) / static_cast<double>(attempts));

    // ---- Optional state snapshot ----
    if (!save_state.empty()) {
        tune::save_field_raw(phi, save_state);
    }
}
