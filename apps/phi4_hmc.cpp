// HMC for the phi^4 scalar field on a `ndim`-dimensional hypercubic lattice
// (default ndim = 4; the example sweeps use ndim = 3 for cleaner FSS in the
// 3D Ising universality class).
//
// Output schema:
//  /run@*                — reproducibility metadata stamped by Writer
//  /vars@*               — every --flag the Parser resolved
//  /therm/stats/s        — S_full per thermalisation trajectory
//  /prod/stats/dH        — H_final - H_initial per production trajectory
//  /prod/stats/accepted  — 0/1 acceptance flag
//  /prod/obs/s           — S_full
//  /prod/obs/mag         — |<phi>|           (per-config magnetization magnitude)
//  /prod/obs/mag_sq      — (<phi>)^2         (per-config magnetization squared; chi input)
//  /prod/obs/m2          — <phi^2>           (per-site field-squared average)

#include <reticolo/reticolo.hpp>

#include <array>
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
    cli::Parser p{"phi4_hmc", "Hybrid Monte Carlo for the phi^4 scalar field"};
    auto const cf      = app::common_flags(p, {.out = "phi4.h5"});
    auto const& kappa  = p.opt<double>("kappa", 0.18, "hopping parameter");
    auto const& lambda = p.opt<double>("lambda", 1.0, "quartic coupling");
    auto const& ndim   = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const rf      = app::hmc_run_flags(p);
    if (!p.parse(argc, argv))
        return 0;

    // ---- State: lattice, action ----
    Lattice<double>::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(cf.L));
    Lattice<double> phi{shape};
    bool const resuming = rf.resuming();

    act::Phi4<double> phi4{.kappa = kappa, .lambda = lambda};

    // ---- Output: writer + series ----
    io::Writer out            = app::open_writer(p, cf, argc, argv);
    std::string const outpath = app::out_path(cf);
    log::act(phi4);
    if (!resuming) {
        out.start_phase("therm");
    }
    out.start_phase("prod");
    auto s_therm  = resuming ? io::Series<double>{} : out.series<double>("/therm/stats/s");
    auto d_h      = out.series<double>("/prod/stats/dH");
    auto accepted = out.series<int>("/prod/stats/accepted");
    auto s_prod   = out.series<double>("/prod/obs/s");
    auto mag      = out.series<double>("/prod/obs/mag");
    auto mag_sq   = out.series<double>("/prod/obs/mag_sq");
    auto m_sq     = out.series<double>("/prod/obs/m2");

    // ---- Updater: owns the RNG (one site stream per canonical slab) ----
    updater::Hmc hmc{phi4, phi, FastRng{cf.seed}, {.tau = rf.tau, .n_md = rf.n_md}};
    long long const start_i = app::resume_or_start(rf, phi, hmc, shape);

    // ---- Thermalisation ----
    if (!resuming) {
        log::info("hmc", "therm  {} trajectories", rf.n_therm);
        for (int i = 0; i < rf.n_therm; ++i) {
            (void)hmc.step(log::Mode::silent);
            s_therm.append(phi4.s_full(phi));
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
            auto const [s1, s2] = obs::reduce(phi, obs::kernel::phi, obs::kernel::phi_sq);
            s_prod.append(phi4.s_full(phi));
            mag.append(obs::mag_abs_of(s1, V));
            mag_sq.append(obs::sq_of_mean_of(s1, V));
            m_sq.append(obs::mean_of(s2, V));
        }
        if (rf.checkpoint_every > 0 && (i + 1) % rf.checkpoint_every == 0) {
            io::save_config(cfg_path(outpath, i + 1), phi, hmc.rng(), i + 1, argc, argv, &p);
        }
    }
}
