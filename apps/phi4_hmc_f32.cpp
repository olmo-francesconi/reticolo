// Single-precision HMC for the phi^4 scalar field — the mixed-precision
// counterpart of phi4_hmc. The field, momentum, force and MD integration all
// run in `float` (half the memory traffic, double-width SIMD), while the action
// reduction S and the Hamiltonian / ΔH are accumulated in `double` so the
// acceptance test stays trustworthy. Precision is derived entirely from the
// lattice type passed to the updater: swap `Lattice<float>` for
// `Lattice<double>` and the same code is full double precision.
//
// Output schema:
//  /run@*                — reproducibility metadata stamped by Writer
//  /vars@*               — every --flag the Parser resolved
//  /therm/stats/s        — S_full per thermalisation trajectory (double)
//  /prod/stats/dH        — H_final - H_initial per production trajectory
//  /prod/stats/accepted  — 0/1 acceptance flag
//  /prod/obs/s           — S_full
//  /prod/obs/mag         — |<phi>|
//  /prod/obs/m2          — <phi^2>

#include <reticolo/reticolo.hpp>

#include <cstddef>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;

    // ---- CLI ----
    cli::Parser p{"phi4_hmc_f32", "Single-precision (mixed) HMC for the phi^4 scalar field"};
    auto const& L          = p.opt<int>("L,size", 8, "linear lattice extent");
    auto const& kappa      = p.opt<double>("kappa", 0.18, "hopping parameter");
    auto const& lambda     = p.opt<double>("lambda", 1.0, "quartic coupling");
    auto const& ndim       = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& tau        = p.opt<double>("tau", 1.0, "HMC trajectory length");
    auto const& n_md       = p.opt<int>("n_md", 20, "MD steps per trajectory");
    auto const& n_therm    = p.opt<int>("n_therm", 200, "thermalisation trajectories");
    auto const& n_prod     = p.opt<int>("n_prod", 1000, "production trajectories");
    auto const& meas_every = p.opt<int>("meas_every", 1, "measure every N trajectories");
    auto const& seed       = p.opt<unsigned long long>("seed", 42ULL, "RNG seed");
    auto const& outpath = p.opt<std::string>("out", std::string{"phi4_f32.h5"}, "HDF5 output path");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    log::start(outpath);

    // ---- State: float lattice, RNG, float action ----
    Lattice<float>::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(L));
    Lattice<float> phi{shape};
    FastRng rng{seed};

    act::Phi4<float> phi4{.kappa = static_cast<float>(kappa), .lambda = static_cast<float>(lambda)};
    log::act(phi4);

    // ---- Output: writer + series (all observables are double) ----
    io::Writer out{outpath, argc, argv, &p};
    out.start_phase("therm");
    out.start_phase("prod");
    auto s_therm  = out.series<double>("/therm/stats/s");
    auto d_h      = out.series<double>("/prod/stats/dH");
    auto accepted = out.series<int>("/prod/stats/accepted");
    auto s_prod   = out.series<double>("/prod/obs/s");
    auto mag      = out.series<double>("/prod/obs/mag");
    auto m_sq     = out.series<double>("/prod/obs/m2");

    // ---- Updater (precision deduced from the float lattice + action) ----
    alg::Hmc hmc{phi4, phi, rng, {.tau = tau, .n_md = n_md}};

    // ---- Thermalisation ----
    log::info("hmc", "therm  {} trajectories", n_therm);
    for (int i = 0; i < n_therm; ++i) {
        (void)hmc.step(log::Mode::silent);
        s_therm.append(phi4.s_full(phi));
    }

    // ---- Production ----
    log::info("hmc", "prod   {} trajectories", n_prod);
    for (int i = 0; i < n_prod; ++i) {
        auto const step = hmc.step();
        d_h.append(step.dH);
        accepted.append(step.accepted ? 1 : 0);
        if (i % meas_every == 0) {
            s_prod.append(phi4.s_full(phi));
            mag.append(obs::mag::abs(phi));
            m_sq.append(obs::sq(phi));
        }
    }
}
