// HMC for the phi^6 scalar field on a 4D hypercubic lattice.
//
// Output schema:
//   /run@*                — reproducibility metadata stamped by Writer
//   /vars@*               — every --flag the Parser resolved
//   /therm/stats/s        — S_full per thermalisation trajectory
//   /prod/stats/dH        — H_final - H_initial per production trajectory
//   /prod/stats/accepted  — 0/1 acceptance flag
//   /prod/obs/s           — S_full
//   /prod/obs/mag         — |<phi>|
//   /prod/obs/m2          — <phi^2>

#include <reticolo/reticolo.hpp>

#include <cstddef>

int main(int argc, char** argv) {
    using namespace reticolo;

    cli::Parser p{"phi6_hmc", "Hybrid Monte Carlo for the phi^6 scalar field"};
    auto const& L          = p.req<int>("L,size", "linear lattice extent");
    auto const& kappa      = p.req<double>("kappa", "hopping parameter");
    auto const& lambda     = p.req<double>("lambda", "quartic coupling");
    auto const& g6         = p.req<double>("g6", "sextic coupling");
    auto const& ndim       = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& tau        = p.opt<double>("tau", 1.0, "HMC trajectory length");
    auto const& n_md       = p.opt<int>("n_md", 20, "MD steps per trajectory");
    auto const& n_therm    = p.opt<int>("n_therm", 200, "thermalisation trajectories");
    auto const& n_prod     = p.opt<int>("n_prod", 1000, "production trajectories");
    auto const& meas_every = p.opt<int>("meas_every", 1, "measure every N trajectories");
    auto const& seed       = p.opt<unsigned long long>("seed", 42ULL, "RNG seed");
    auto const& outpath    = p.opt<std::string>("out", std::string{"phi6.h5"}, "HDF5 output path");
    if (!p.parse(argc, argv))
        return 0;

    log::start(outpath);

    Lattice<double>::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(L));
    Lattice<double> phi{shape};
    FastRng rng{seed};
    act::Phi6<double> phi6{.kappa = kappa, .lambda = lambda, .g6 = g6};
    log::act(phi6);

    io::Writer out{outpath, argc, argv, &p};
    out.start_phase("therm");
    out.start_phase("prod");
    auto s_therm  = out.series<double>("/therm/stats/s");
    auto d_h      = out.series<double>("/prod/stats/dH");
    auto accepted = out.series<int>("/prod/stats/accepted");
    auto s_prod   = out.series<double>("/prod/obs/s");
    auto mag      = out.series<double>("/prod/obs/mag");
    auto m_sq     = out.series<double>("/prod/obs/m2");

    alg::Hmc<act::Phi6<double>, FastRng> hmc{phi6, phi, rng, {.tau = tau, .n_md = n_md}};
    log::algo(hmc);

    log::info("hmc", "therm  {} trajectories", n_therm);
    for (int i = 0; i < n_therm; ++i) {
        (void)hmc.trajectory(log::Mode::silent);
        s_therm.append(phi6.s_full(phi));
    }
    log::info("hmc", "prod   {} trajectories", n_prod);
    for (int i = 0; i < n_prod; ++i) {
        auto const step = hmc.trajectory();
        d_h.append(step.dH);
        accepted.append(step.accepted ? 1 : 0);
        if (i % meas_every == 0) {
            s_prod.append(phi6.s_full(phi));
            mag.append(obs::magnetization(phi));
            m_sq.append(obs::m2(phi));
        }
    }
}
