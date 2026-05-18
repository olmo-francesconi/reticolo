// Phase-quenched HMC for the 4D self-interacting relativistic lattice Bose
// gas at finite chemical potential. Samples the real (phase-quenched) part
// `S_R = base.s_full(phi)`; observes both S_R and the imaginary observable
// `S_I = base.s_imag(phi)` per trajectory. Output is the reference
// distribution that the LLR run reconstructs.

#include <reticolo/reticolo.hpp>

#include <complex>
#include <cstddef>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Action = act::BoseGas<double>;

    cli::Parser p{"bose_gas_hmc", "Phase-quenched HMC for the 4D Bose gas (records S_R + S_I)"};
    auto const& L          = p.req<int>("L,size", "linear lattice extent");
    auto const& ndim       = p.opt<int>("ndim", 4, "spacetime dimensions");
    auto const& mass       = p.opt<double>("mass", 1.0, "bare mass m");
    auto const& lambda     = p.opt<double>("lambda", 1.0, "quartic coupling lambda");
    auto const& mu         = p.req<double>("mu", "chemical potential mu");
    auto const& tau        = p.opt<double>("tau", 1.0, "HMC trajectory length");
    auto const& n_md       = p.opt<int>("n_md", 10, "MD steps per trajectory");
    auto const& n_therm    = p.opt<int>("n_therm", 500, "thermalisation trajectories");
    auto const& n_prod     = p.opt<int>("n_prod", 5000, "production trajectories");
    auto const& meas_every = p.opt<int>("meas_every", 1, "measure every N trajectories");
    auto const& seed       = p.opt<unsigned long long>("seed", 42ULL, "RNG seed");
    auto const& outpath =
        p.opt<std::string>("out", std::string{"bose_gas_hmc.h5"}, "HDF5 output path");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    Lattice<std::complex<double>>::SizeVec shape(static_cast<std::size_t>(ndim),
                                                 static_cast<std::size_t>(L));
    Lattice<std::complex<double>> phi{shape};
    FastRng rng{seed};
    Action const action{.mass = mass, .lambda = lambda, .mu = mu};

    io::Writer out{outpath, argc, argv, &p};
    out.start_phase("therm");
    out.start_phase("prod");
    auto s_r_therm = out.series<double>("/therm/stats/s_r");
    auto d_h       = out.series<double>("/prod/stats/dH");
    auto accepted  = out.series<int>("/prod/stats/accepted");
    auto s_r       = out.series<double>("/prod/obs/s_r");
    auto s_i       = out.series<double>("/prod/obs/s_i");

    alg::Hmc<Action, FastRng, alg::integ::Omelyan2> hmc{
        action, phi, rng, {.tau = tau, .n_md = n_md}};

    for (int i = 0; i < n_therm; ++i) {
        (void)hmc.trajectory();
        s_r_therm.append(action.s_full(phi));
    }
    for (int i = 0; i < n_prod; ++i) {
        auto const step = hmc.trajectory();
        d_h.append(step.dH);
        accepted.append(step.accepted ? 1 : 0);
        if (i % meas_every == 0) {
            s_r.append(action.s_full(phi));
            s_i.append(action.s_imag(phi));
        }
    }
}
