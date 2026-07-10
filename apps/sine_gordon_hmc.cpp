// HMC for the sine-Gordon scalar field on a 4D hypercubic lattice.
//
//  S = sum_x [ -2 kappa phi(x) Σ_{mu>0} phi(x+mu)  +  phi(x)^2
//              - alpha cos(phi(x)) ]
//
// Output schema:
//  /run@*                — reproducibility metadata stamped by Writer
//  /vars@*               — every --flag the Parser resolved
//  /therm/stats/s        — S_full per thermalisation trajectory
//  /prod/stats/dH        — H_final - H_initial per production trajectory
//  /prod/stats/accepted  — 0/1 acceptance flag
//  /prod/obs/s           — S_full
//  /prod/obs/mag         — |<phi>|
//  /prod/obs/cos_phi     — <cos(phi)>

#include <reticolo/reticolo.hpp>

#include <cmath>
#include <cstddef>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;

    // ---- CLI ----
    cli::Parser p{"sine_gordon_hmc", "HMC for the sine-Gordon scalar field"};
    auto const cf          = app::common_flags(p, {.out = "sine_gordon.h5"});
    auto const& kappa      = p.opt<double>("kappa", 1.0, "hopping parameter");
    auto const& alpha      = p.opt<double>("alpha", 1.0, "cosine-potential strength");
    auto const& ndim       = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& tau        = p.opt<double>("tau", 1.0, "HMC trajectory length");
    auto const& n_md       = p.opt<int>("n_md", 20, "MD steps per trajectory");
    auto const& n_therm    = p.opt<int>("n_therm", 200, "thermalisation trajectories");
    auto const& n_prod     = p.opt<int>("n_prod", 1000, "production trajectories");
    auto const& meas_every = p.opt<int>("meas_every", 1, "measure every N trajectories");
    if (!p.parse(argc, argv))
        return 0;

    io::Writer out = app::open_writer(p, cf, argc, argv);

    // ---- State: lattice, RNG, action ----
    Lattice<double>::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(cf.L));
    Lattice<double> phi{shape};
    act::SineGordon<double> sg{.kappa = kappa, .alpha = alpha};
    log::act(sg);

    // ---- Output: writer + series ----
    out.start_phase("therm");
    out.start_phase("prod");
    auto s_therm  = out.series<double>("/therm/stats/s");
    auto d_h      = out.series<double>("/prod/stats/dH");
    auto accepted = out.series<int>("/prod/stats/accepted");
    auto s_prod   = out.series<double>("/prod/obs/s");
    auto mag      = out.series<double>("/prod/obs/mag");
    auto cos_phi  = out.series<double>("/prod/obs/cos_phi");

    // ---- Updater ----
    alg::Hmc hmc{sg, phi, FastRng{cf.seed}, {.tau = tau, .n_md = n_md}};

    auto cos_avg = [&phi]() {
        double sum = 0.0;
        for (Site const x : phi.sites()) {
            sum += std::cos(phi[x]);
        }
        return sum / static_cast<double>(phi.nsites());
    };

    // ---- Thermalisation ----
    log::info("hmc", "therm  {} trajectories", n_therm);
    for (int i = 0; i < n_therm; ++i) {
        (void)hmc.step(log::Mode::silent);
        s_therm.append(sg.s_full(phi));
    }

    // ---- Production ----
    log::info("hmc", "prod   {} trajectories", n_prod);
    for (int i = 0; i < n_prod; ++i) {
        auto const step = hmc.step();
        d_h.append(step.dH);
        accepted.append(step.accepted ? 1 : 0);
        if (i % meas_every == 0) {
            s_prod.append(sg.s_full(phi));
            mag.append(obs::mag::abs(phi));
            cos_phi.append(cos_avg());
        }
    }
}
