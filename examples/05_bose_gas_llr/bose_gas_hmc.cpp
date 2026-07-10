// Phase-quenched HMC for the 4D self-interacting relativistic lattice Bose
// gas at finite chemical potential. Samples the real (phase-quenched) part
// `S_R = base.s_full(phi)`; observes both S_R and the imaginary observable
// `S_I = base.s_imag(phi)` per trajectory. Output is the reference
// distribution that the LLR run reconstructs.

#include <reticolo/reticolo.hpp>

#include <complex>
#include <cstddef>
#include <filesystem>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Action = act::BoseGas<double>;

    // ---- CLI ----
    cli::Parser p{"bose_gas_hmc", "Phase-quenched HMC for the 4D Bose gas (records S_R + S_I)"};
    auto const& L          = p.opt<int>("L,size", 4, "linear lattice extent");
    auto const& ndim       = p.opt<int>("ndim", 4, "spacetime dimensions");
    auto const& mass       = p.opt<double>("mass", 1.0, "bare mass m");
    auto const& lambda     = p.opt<double>("lambda", 1.0, "quartic coupling lambda");
    auto const& mu         = p.opt<double>("mu", 1.0, "chemical potential mu");
    auto const& tau        = p.opt<double>("tau", 1.0, "HMC trajectory length");
    auto const& n_md       = p.opt<int>("n_md", 10, "MD steps per trajectory");
    auto const& n_therm    = p.opt<int>("n_therm", 500, "thermalisation trajectories");
    auto const& n_prod     = p.opt<int>("n_prod", 5000, "production trajectories");
    auto const& meas_every = p.opt<int>("meas_every", 1, "measure every N trajectories");
    auto const& seed       = p.opt<unsigned long long>("seed", 42ULL, "RNG seed");
    auto const& workspace =
        p.opt<std::string>("workspace", std::string{"."}, "workspace folder (output + logs)");
    auto const& outfile = p.opt<std::string>(
        "out", std::string{"bose_gas_hmc.h5"}, "HDF5 output file name, inside workspace");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    log::start(workspace, outfile);
    std::string const outpath = (std::filesystem::path{workspace} / outfile).string();

    // ---- State: lattice, RNG, action ----
    Lattice<std::complex<double>>::SizeVec shape(static_cast<std::size_t>(ndim),
                                                 static_cast<std::size_t>(L));
    Lattice<std::complex<double>> phi{shape};
    Action const action{.mass = mass, .lambda = lambda, .mu = mu};
    log::act(action);

    // ---- Output: writer + series ----
    io::Writer out{outpath, argc, argv, &p};
    out.start_phase("therm");
    out.start_phase("prod");
    auto s_therm  = out.series<double>("/therm/stats/s");
    auto d_h      = out.series<double>("/prod/stats/dH");
    auto accepted = out.series<int>("/prod/stats/accepted");
    auto s_r      = out.series<double>("/prod/obs/s_r");
    auto s_i      = out.series<double>("/prod/obs/s_i");

    // ---- Updater ----
    alg::Hmc hmc{action, phi, FastRng{seed}, {.tau = tau, .n_md = n_md}, alg::integ::omelyan2};

    // ---- Thermalisation ----
    log::info("hmc", "therm  {} trajectories", n_therm);
    for (int i = 0; i < n_therm; ++i) {
        (void)hmc.step(log::Mode::silent);
        s_therm.append(action.s_full(phi));
    }

    // ---- Production ----
    log::info("hmc", "prod   {} trajectories", n_prod);
    for (int i = 0; i < n_prod; ++i) {
        auto const step = hmc.step();
        d_h.append(step.dH);
        accepted.append(step.accepted ? 1 : 0);
        if (i % meas_every == 0) {
            s_r.append(action.s_full(phi));
            s_i.append(action.s_imag(phi));
        }
    }
}
