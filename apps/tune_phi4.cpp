// Single-binary parameter-tuning rig for Phi4. Drives ONE of:
//
//   --algo metropolis     --sigma <σ>
//   --algo hmc_leapfrog   --tau <τ> --n_md <n>
//   --algo hmc_omelyan2   --tau <τ> --n_md <n>
//   --algo hmc_omelyan4   --tau <τ> --n_md <n>
//
// Dumps per-update time series of S and Σφ² (obs::sq_of_mean) to HDF5, with the
// wall-time of the production loop stamped as /prod@wall_seconds so the
// Python autocorrelation analysis can report τ_int in actual seconds rather
// than algorithm-internal "steps".
//
// Output schema:
//   /run@*                      reproducibility metadata
//   /vars@*                     every --flag the Parser resolved
//   /prod@wall_seconds          inner-loop wall time (seconds, double)
//   /prod@n_meas                number of production measurements (int)
//   /prod@accept_rate           acceptance over the production loop (double)
//   /prod/obs/s                 S_full per update
//   /prod/obs/mean_sq           Σφ² / V per update

#include <reticolo/reticolo.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace {

using bench_clock = std::chrono::steady_clock;

template <class Integrator>
void run_hmc(reticolo::act::Phi4<double> const& action,
             reticolo::Lattice<double>& field,
             reticolo::FastRng& rng,
             double tau,
             int n_md,
             int n_therm,
             int n_prod,
             reticolo::io::Writer& out) {
    using namespace reticolo;

    alg::Hmc<act::Phi4<double>, FastRng, Integrator> hmc{
        action, field, rng, {.tau = tau, .n_md = n_md}};

    for (int i = 0; i < n_therm; ++i) {
        (void)hmc.trajectory();
    }

    auto s       = out.series<double>("/prod/obs/s");
    auto mean_sq = out.series<double>("/prod/obs/mean_sq");

    int accepted       = 0;
    double algo_s      = 0.0;  // sum of algorithm-only wall time across all updates
    double obs_s       = 0.0;
    auto const t_wall0 = bench_clock::now();
    for (int i = 0; i < n_prod; ++i) {
        auto const ta   = bench_clock::now();
        auto const step = hmc.trajectory();
        auto const tb   = bench_clock::now();
        algo_s += std::chrono::duration<double>(tb - ta).count();
        accepted += step.accepted ? 1 : 0;

        auto const oa = bench_clock::now();
        s.append(action.s_full(field));
        mean_sq.append(obs::sq_of_mean(field));
        auto const ob = bench_clock::now();
        obs_s += std::chrono::duration<double>(ob - oa).count();
    }
    double const wall_s = std::chrono::duration<double>(bench_clock::now() - t_wall0).count();

    out.attr<double>("/prod@wall_seconds", wall_s);
    out.attr<double>("/prod@algo_seconds", algo_s);
    out.attr<double>("/prod@obs_seconds", obs_s);
    out.attr<int>("/prod@n_meas", n_prod);
    out.attr<double>("/prod@accept_rate",
                     static_cast<double>(accepted) / static_cast<double>(n_prod));
}

void run_metropolis(reticolo::act::Phi4<double> const& action,
                    reticolo::Lattice<double>& field,
                    reticolo::FastRng& rng,
                    double sigma,
                    int n_therm,
                    int n_prod,
                    reticolo::io::Writer& out) {
    using namespace reticolo;

    alg::Metropolis<act::Phi4<double>, FastRng> mc{
        action, field, rng, alg::MetropolisSpec{.sigma = sigma}};

    for (int i = 0; i < n_therm; ++i) {
        (void)mc.sweep();
    }

    auto s       = out.series<double>("/prod/obs/s");
    auto mean_sq = out.series<double>("/prod/obs/mean_sq");

    std::size_t accepted = 0;
    std::size_t attempts = 0;
    double algo_s        = 0.0;
    double obs_s         = 0.0;
    auto const t_wall0   = bench_clock::now();
    for (int i = 0; i < n_prod; ++i) {
        auto const ta    = bench_clock::now();
        auto const stats = mc.sweep();
        auto const tb    = bench_clock::now();
        algo_s += std::chrono::duration<double>(tb - ta).count();
        accepted += stats.accepted;
        attempts += stats.attempts;

        auto const oa = bench_clock::now();
        s.append(action.s_full(field));
        mean_sq.append(obs::sq_of_mean(field));
        auto const ob = bench_clock::now();
        obs_s += std::chrono::duration<double>(ob - oa).count();
    }
    double const wall_s = std::chrono::duration<double>(bench_clock::now() - t_wall0).count();

    out.attr<double>("/prod@wall_seconds", wall_s);
    out.attr<double>("/prod@algo_seconds", algo_s);
    out.attr<double>("/prod@obs_seconds", obs_s);
    out.attr<int>("/prod@n_meas", n_prod);
    out.attr<double>("/prod@accept_rate",
                     attempts == 0 ? 0.0
                                   : static_cast<double>(accepted) / static_cast<double>(attempts));
}

}  // namespace

int main(int argc, char** argv) {
    using namespace reticolo;

    // Tuning rig: trajectory wall time is the signal — silence per-step
    // logger output so the format work doesn't pollute the measurement.
    log::off();

    cli::Parser p{"tune_phi4", "Algorithm-tuning rig for Phi4 (autocorrelation vs wall time)"};
    auto const& L          = p.opt<int>("L,size", 8, "linear lattice extent");
    auto const& kappa      = p.opt<double>("kappa", 0.18, "hopping parameter");
    auto const& lambda     = p.opt<double>("lambda", 1.0, "quartic coupling");
    auto const& ndim       = p.opt<int>("ndim", 3, "spatial dimensions");
    auto const& algo       = p.opt<std::string>("algo",
                                          std::string{"hmc_leapfrog"},
                                          "one of: metropolis, hmc_leapfrog, "
                                                "hmc_omelyan2, hmc_omelyan4");
    auto const& sigma      = p.opt<double>("sigma", 0.4, "Metropolis proposal width");
    auto const& tau        = p.opt<double>("tau", 1.0, "HMC trajectory length");
    auto const& n_md       = p.opt<int>("n_md", 20, "MD steps per HMC trajectory");
    auto const& n_therm    = p.opt<int>("n_therm", 1000, "thermalisation updates");
    auto const& n_prod     = p.opt<int>("n_prod", 5000, "production updates");
    auto const& seed       = p.opt<unsigned long long>("seed", 42ULL, "RNG seed");
    auto const& outpath    = p.opt<std::string>("out", std::string{"tune.h5"}, "HDF5 output path");
    auto const& init_from  = p.opt<std::string>("init_from",
                                               std::string{},
                                               "raw-binary field state to load before "
                                                "production (skip thermalisation)");
    auto const& save_state = p.opt<std::string>("save_state",
                                                std::string{},
                                                "raw-binary field state to save at the "
                                                "end of the run (after production)");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    Lattice<double>::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(L));
    Lattice<double> phi{shape};
    FastRng rng{seed};
    act::Phi4<double> const phi4{.kappa = kappa, .lambda = lambda};

    // Load thermalised state from a previous run (shared across the parameter
    // grid). Layout: `nsites()` doubles in lattice index order. Caller must
    // pass --n_therm=0 when using this — we don't enforce it but the warmup
    // will then overwrite the loaded state.
    if (!init_from.empty()) {
        std::FILE* f = std::fopen(init_from.c_str(), "rb");
        if (f == nullptr) {
            std::fprintf(stderr, "init_from: cannot open %s\n", init_from.c_str());
            return 1;
        }
        std::size_t const got = std::fread(phi.data(), sizeof(double), phi.nsites(), f);
        std::fclose(f);
        if (got != phi.nsites()) {
            std::fprintf(stderr,
                         "init_from: expected %zu doubles, got %zu — shape mismatch?\n",
                         phi.nsites(),
                         got);
            return 1;
        }
    }

    io::Writer out{outpath, argc, argv, &p};
    out.start_phase("prod");

    if (algo == "metropolis") {
        run_metropolis(phi4, phi, rng, sigma, n_therm, n_prod, out);
    } else if (algo == "hmc_leapfrog") {
        run_hmc<alg::integ::Leapfrog>(phi4, phi, rng, tau, n_md, n_therm, n_prod, out);
    } else if (algo == "hmc_omelyan2") {
        run_hmc<alg::integ::Omelyan2>(phi4, phi, rng, tau, n_md, n_therm, n_prod, out);
    } else if (algo == "hmc_omelyan4") {
        run_hmc<alg::integ::Omelyan4>(phi4, phi, rng, tau, n_md, n_therm, n_prod, out);
    } else {
        std::fprintf(stderr, "unknown --algo: %s\n", algo.c_str());
        return 1;
    }

    if (!save_state.empty()) {
        std::FILE* f = std::fopen(save_state.c_str(), "wb");
        if (f == nullptr) {
            std::fprintf(stderr, "save_state: cannot open %s for writing\n", save_state.c_str());
            return 1;
        }
        std::fwrite(phi.data(), sizeof(double), phi.nsites(), f);
        std::fclose(f);
    }
}
