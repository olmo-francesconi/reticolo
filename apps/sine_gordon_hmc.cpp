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
    cli::Parser p{"sine_gordon_hmc", "HMC for the sine-Gordon scalar field"};
    auto const cf     = app::common_flags(p, {.out = "sine_gordon.h5"});
    auto const& kappa = p.opt<double>("kappa", 1.0, "hopping parameter");
    auto const& alpha = p.opt<double>("alpha", 1.0, "cosine-potential strength");
    auto const& ndim  = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const rf     = app::hmc_run_flags(p);
    if (!p.parse(argc, argv))
        return 0;

    // ---- State: lattice, action ----
    Lattice<double>::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(cf.L));
    Lattice<double> phi{shape};
    bool const resuming = rf.resuming();

    act::SineGordon<double> sg{.kappa = kappa, .alpha = alpha};

    // ---- Output: writer + series ----
    io::Writer out            = app::open_writer(p, cf, argc, argv);
    std::string const outpath = app::out_path(cf);
    log::act(sg);
    if (!resuming) {
        out.start_phase("therm");
    }
    out.start_phase("prod");
    auto s_therm  = resuming ? io::Series<double>{} : out.series<double>("/therm/stats/s");
    auto d_h      = out.series<double>("/prod/stats/dH");
    auto accepted = out.series<int>("/prod/stats/accepted");
    auto s_prod   = out.series<double>("/prod/obs/s");
    auto mag      = out.series<double>("/prod/obs/mag");
    auto cos_phi  = out.series<double>("/prod/obs/cos_phi");

    // ---- Updater ----
    updater::Hmc hmc{sg, phi, FastRng{cf.seed}, {.tau = rf.tau, .n_md = rf.n_md}};
    long long const start_i = app::resume_or_start(rf, phi, hmc, shape);

    // ---- Thermalisation ----
    if (!resuming) {
        log::info("hmc", "therm  {} trajectories", rf.n_therm);
        for (int i = 0; i < rf.n_therm; ++i) {
            (void)hmc.step(log::Mode::silent);
            s_therm.append(sg.s_full(phi));
        }
    }

    // ---- Production ----
    log::info("hmc", "prod   {} trajectories (from {})", rf.n_prod, start_i);
    for (long long i = start_i; i < rf.n_prod; ++i) {
        auto const step = hmc.step();
        d_h.append(step.dH);
        accepted.append(step.accepted ? 1 : 0);
        if (i % rf.meas_every == 0) {
            auto const V        = static_cast<double>(phi.nsites());
            auto const [s1, sc] = obs::reduce(
                phi, obs::kernel::phi, [](auto z) { return std::cos(static_cast<double>(z)); });
            s_prod.append(sg.s_full(phi));
            mag.append(obs::mag_abs_of(s1, V));
            cos_phi.append(obs::mean_of(sc, V));
        }
        if (rf.checkpoint_every > 0 && (i + 1) % rf.checkpoint_every == 0) {
            io::save_config(cfg_path(outpath, i + 1), phi, hmc.rng(), i + 1, argc, argv, &p);
        }
    }
}
