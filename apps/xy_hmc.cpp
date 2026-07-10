// HMC for the XY (planar rotor) model on a hypercubic lattice.
//
//   S = -beta * sum_<x,y> cos(theta(x) - theta(y))
//
// theta(x) is an angle stored in a plain scalar Lattice<double>; the default
// (all-zero) field is the cold, fully-aligned start.
//
// Output schema:
//  /run@*                — reproducibility metadata stamped by Writer
//  /vars@*               — every --flag the Parser resolved
//  /therm/stats/s        — S_full per thermalisation trajectory
//  /prod/stats/dH        — H_final - H_initial per production trajectory
//  /prod/stats/accepted  — 0/1 acceptance flag
//  /prod/obs/s           — S_full
//  /prod/obs/mag         — |<(cos θ, sin θ)>|   (XY vector magnetization)

#include <reticolo/reticolo.hpp>

#include <cmath>
#include <cstddef>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;

    // ---- CLI ----
    cli::Parser p{"xy_hmc", "Hybrid Monte Carlo for the XY (planar rotor) model"};
    auto const cf          = app::common_flags(p, {.out = "xy.h5"});
    auto const& beta       = p.opt<double>("beta", 0.45, "inverse-temperature coupling");
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

    // ---- State: lattice, RNG, action ----
    Lattice<double>::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(cf.L));
    Lattice<double> theta{shape};
    act::Xy<double> xy{.beta = beta};
    log::act(xy);

    // ---- Output: writer + series ----
    out.start_phase("therm");
    out.start_phase("prod");
    auto s_therm  = out.series<double>("/therm/stats/s");
    auto d_h      = out.series<double>("/prod/stats/dH");
    auto accepted = out.series<int>("/prod/stats/accepted");
    auto s_prod   = out.series<double>("/prod/obs/s");
    auto mag      = out.series<double>("/prod/obs/mag");

    // ---- Updater ----
    alg::Hmc hmc{xy, theta, FastRng{cf.seed}, {.tau = tau, .n_md = n_md}};

    // |<(cos θ, sin θ)>| — the XY order parameter (0 disordered, 1 aligned).
    auto xy_mag = [&theta]() {
        double cx = 0.0;
        double cy = 0.0;
        for (Site const x : theta.sites()) {
            cx += std::cos(theta[x]);
            cy += std::sin(theta[x]);
        }
        double const inv = 1.0 / static_cast<double>(theta.nsites());
        return std::hypot(cx * inv, cy * inv);
    };

    // ---- Thermalisation ----
    log::info("hmc", "therm  {} trajectories", n_therm);
    for (int i = 0; i < n_therm; ++i) {
        (void)hmc.step(log::Mode::silent);
        s_therm.append(xy.s_full(theta));
    }

    // ---- Production ----
    log::info("hmc", "prod   {} trajectories", n_prod);
    for (int i = 0; i < n_prod; ++i) {
        auto const step = hmc.step();
        d_h.append(step.dH);
        accepted.append(step.accepted ? 1 : 0);
        if (i % meas_every == 0) {
            s_prod.append(xy.s_full(theta));
            mag.append(xy_mag());
        }
    }
}
