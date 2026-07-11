// Single-precision (mixed) HMC for the phi^6 scalar field — the mixed-precision
// counterpart of phi6_hmc. The field, momentum, force and MD integration run in
// `float`; the action reduction S and the Hamiltonian / ΔH accumulate in
// `double` so the acceptance test stays trustworthy. Precision is derived from
// the lattice type: swap `Lattice<float>` for `Lattice<double>` for full double.
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
    cli::Parser p{"phi6_hmc_f32", "Single-precision (mixed) HMC for the phi^6 scalar field"};
    auto const cf          = app::common_flags(p, {.out = "phi6_f32.h5"});
    auto const& kappa      = p.opt<double>("kappa", 0.13, "hopping parameter");
    auto const& lambda     = p.opt<double>("lambda", 0.05, "quartic coupling");
    auto const& g6         = p.opt<double>("g6", 0.01, "sextic coupling");
    auto const& ndim       = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& tau        = p.opt<double>("tau", 1.0, "HMC trajectory length");
    auto const& n_md       = p.opt<int>("n_md", 20, "MD steps per trajectory");
    auto const& n_therm    = p.opt<int>("n_therm", 200, "thermalisation trajectories");
    auto const& n_prod     = p.opt<int>("n_prod", 1000, "production trajectories");
    auto const& meas_every = p.opt<int>("meas_every", 1, "measure every N trajectories");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    io::Writer out = app::open_writer(p, cf, argc, argv);

    // ---- State: float lattice, RNG, float action ----
    Lattice<float>::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(cf.L));
    Lattice<float> phi{shape};
    act::Phi6<float> phi6{.kappa  = static_cast<float>(kappa),
                          .lambda = static_cast<float>(lambda),
                          .g6     = static_cast<float>(g6)};
    log::act(phi6);

    // ---- Output: writer + series (all observables are double) ----
    out.start_phase("therm");
    out.start_phase("prod");
    auto s_therm  = out.series<double>("/therm/stats/s");
    auto d_h      = out.series<double>("/prod/stats/dH");
    auto accepted = out.series<int>("/prod/stats/accepted");
    auto s_prod   = out.series<double>("/prod/obs/s");
    auto mag      = out.series<double>("/prod/obs/mag");
    auto m_sq     = out.series<double>("/prod/obs/m2");

    // ---- Updater (precision deduced from the float lattice + action) ----
    updater::Hmc hmc{phi6, phi, FastRng{cf.seed}, {.tau = tau, .n_md = n_md}};

    // ---- Thermalisation ----
    log::info("hmc", "therm  {} trajectories", n_therm);
    for (int i = 0; i < n_therm; ++i) {
        (void)hmc.step(log::Mode::silent);
        s_therm.append(phi6.s_full(phi));
    }

    // ---- Production ----
    log::info("hmc", "prod   {} trajectories", n_prod);
    for (int i = 0; i < n_prod; ++i) {
        auto const step = hmc.step();
        d_h.append(step.dH);
        accepted.append(step.accepted ? 1 : 0);
        if (i % meas_every == 0) {
            s_prod.append(phi6.s_full(phi));
            mag.append(obs::mag::abs(phi));
            m_sq.append(obs::sq(phi));
        }
    }
}
