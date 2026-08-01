// Single-precision (mixed) HMC for the XY (planar rotor) model — the
// mixed-precision counterpart of xy_hmc. The angle field, momentum, force and
// MD integration run in `float`; the action reduction S and the Hamiltonian /
// ΔH accumulate in `double`. Precision is derived from the lattice type: swap
// `Lattice<float>` for `Lattice<double>` for full double.
//
//   S = -beta * sum_<x,y> cos(theta(x) - theta(y))
//
// Output schema:
//  /run@*                — reproducibility metadata stamped by Writer
//  /vars@*               — every --flag the Parser resolved
//  /therm/stats/s        — S_full per thermalisation trajectory (double)
//  /prod/stats/dH        — H_final - H_initial per production trajectory
//  /prod/stats/accepted  — 0/1 acceptance flag
//  /prod/obs/s           — S_full
//  /prod/obs/mag         — |<(cos θ, sin θ)>|   (XY vector magnetization)

#include <reticolo/reticolo.hpp>

#include <array>
#include <cmath>
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

    // ---- CLI ----
    cli::Parser p{"xy_hmc_f32", "Single-precision (mixed) HMC for the XY (planar rotor) model"};
    auto const cf    = app::common_flags(p, {.out = "xy_f32.h5"});
    auto const& beta = p.opt<double>("beta", 0.45, "inverse-temperature coupling");
    auto const& ndim = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const rf    = app::hmc_run_flags(p);
    if (!p.parse(argc, argv)) {
        return 0;
    }

    // ---- State: float lattice, action ----
    Lattice<float>::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(cf.L));
    Lattice<float> theta{shape};
    bool const resuming = rf.resuming();

    act::Xy<float> xy{.beta = static_cast<float>(beta)};

    // ---- Output: writer + series (all observables are double) ----
    io::Writer out            = app::open_writer(p, cf, argc, argv);
    std::string const outpath = app::out_path(cf);
    log::act(xy);
    if (!resuming) {
        out.start_phase("therm");
    }
    out.start_phase("prod");
    auto s_therm  = resuming ? io::Series<double>{} : out.series<double>("/therm/stats/s");
    auto d_h      = out.series<double>("/prod/stats/dH");
    auto accepted = out.series<int>("/prod/stats/accepted");
    auto s_prod   = out.series<double>("/prod/obs/s");
    auto mag      = out.series<double>("/prod/obs/mag");

    // ---- Updater (precision deduced from the float lattice + action) ----
    updater::Hmc hmc{xy, theta, FastRng{cf.seed}, {.tau = rf.tau, .n_md = rf.n_md}};
    long long const start_i = app::resume_or_start(rf, theta, hmc, shape);

    // |<(cos θ, sin θ)>| — the XY order parameter (0 disordered, 1 aligned).
    auto xy_mag = [&theta]() {
        double cx = 0.0;
        double cy = 0.0;
        for (Site const x : theta.sites()) {
            cx += std::cos(static_cast<double>(theta[x]));
            cy += std::sin(static_cast<double>(theta[x]));
        }
        double const inv = 1.0 / static_cast<double>(theta.nsites());
        return std::hypot(cx * inv, cy * inv);
    };

    // ---- Thermalisation ----
    if (!resuming) {
        log::info("hmc", "therm  {} trajectories", rf.n_therm);
        for (int i = 0; i < rf.n_therm; ++i) {
            (void)hmc.step(log::Mode::silent);
            s_therm.append(xy.s_full(theta));
        }
    }

    // ---- Production ----
    log::info("hmc", "prod   {} trajectories (from {})", rf.n_prod, start_i);
    for (long long i = start_i; i < rf.n_prod; ++i) {
        auto const step = hmc.step();
        d_h.append(step.dH);
        accepted.append(step.accepted ? 1 : 0);
        if (i % rf.meas_every == 0) {
            s_prod.append(xy.s_full(theta));
            mag.append(xy_mag());
        }
        if (rf.checkpoint_every > 0 && (i + 1) % rf.checkpoint_every == 0) {
            io::save_config(cfg_path(outpath, i + 1), theta, hmc.rng(), i + 1, argc, argv, &p);
        }
    }
}
