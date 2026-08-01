// Phase-quenched HMC for the 4D self-interacting relativistic lattice Bose
// gas at finite chemical potential. Samples the real (phase-quenched) part
// `S_R = base.s_full(phi)`; observes both S_R and the imaginary observable
// `S_I = base.s_imag(phi)` per trajectory. Output is the reference
// distribution that the LLR run reconstructs.

#include <reticolo/reticolo.hpp>

#include <array>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <string>

namespace {

std::string cfg_path(std::string const& out, long long i) {
    std::string stem = out;
    if (auto const pos = stem.rfind(".h5"); pos != std::string::npos && pos == stem.size() - 3) {
        stem.resize(pos);
    }
    std::array<char, 256> buf{};
    std::snprintf(buf.data(), buf.size(), "%s.cfg.%05lld.h5", stem.c_str(), i);
    return buf.data();
}

}  // namespace

int main(int argc, char** argv) {
    using namespace reticolo;
    using Action = act::BoseGas<double>;

    // ---- CLI ----
    cli::Parser p{"bose_gas_hmc", "Phase-quenched HMC for the 4D Bose gas (records S_R + S_I)"};
    auto const cf      = app::common_flags(p, {.L = 4, .out = "bose_gas_hmc.h5"});
    auto const& ndim   = p.opt<int>("ndim", 4, "spacetime dimensions");
    auto const& mass   = p.opt<double>("mass", 1.0, "bare mass m");
    auto const& lambda = p.opt<double>("lambda", 1.0, "quartic coupling lambda");
    auto const& mu     = p.opt<double>("mu", 1.0, "chemical potential mu");
    auto const rf      = app::hmc_run_flags(p, {.n_md = 10, .n_therm = 500, .n_prod = 5000});
    if (!p.parse(argc, argv)) {
        return 0;
    }

    // ---- State: lattice, action ----
    Lattice<std::complex<double>>::SizeVec shape(static_cast<std::size_t>(ndim),
                                                 static_cast<std::size_t>(cf.L));
    Lattice<std::complex<double>> phi{shape};
    bool const resuming = rf.resuming();

    Action const action{.mass = mass, .lambda = lambda, .mu = mu};

    // ---- Output: writer + series ----
    io::Writer out            = app::open_writer(p, cf, argc, argv);
    std::string const outpath = app::out_path(cf);
    log::act(action);
    if (!resuming) {
        out.start_phase("therm");
    }
    out.start_phase("prod");
    auto s_therm  = resuming ? io::Series<double>{} : out.series<double>("/therm/stats/s");
    auto d_h      = out.series<double>("/prod/stats/dH");
    auto accepted = out.series<int>("/prod/stats/accepted");
    auto s_r      = out.series<double>("/prod/obs/s_r");
    auto s_i      = out.series<double>("/prod/obs/s_i");

    // ---- Updater ----
    updater::Hmc hmc{action, phi, FastRng{cf.seed}, {.tau = rf.tau, .n_md = rf.n_md}};
    long long const start_i = app::resume_or_start(rf, phi, hmc, shape);

    // ---- Thermalisation ----
    if (!resuming) {
        log::info("hmc", "therm  {} trajectories", rf.n_therm);
        for (int i = 0; i < rf.n_therm; ++i) {
            (void)hmc.step(log::Mode::silent);
            s_therm.append(action.s_full(phi));
        }
    }

    // ---- Production ----
    log::info("hmc", "prod   {} trajectories (from {})", rf.n_prod, start_i);
    for (long long i = start_i; i < rf.n_prod; ++i) {
        auto const step = hmc.step();
        d_h.append(step.dH);
        accepted.append(step.accepted ? 1 : 0);
        if (i % rf.meas_every == 0) {
            s_r.append(action.s_full(phi));
            s_i.append(action.s_imag(phi));
        }
        if (rf.checkpoint_every > 0 && (i + 1) % rf.checkpoint_every == 0) {
            io::save_config(cfg_path(outpath, i + 1), phi, hmc.rng(), i + 1, argc, argv, &p);
        }
    }
}
