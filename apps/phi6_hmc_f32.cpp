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
    cli::Parser p{"phi6_hmc_f32", "Single-precision (mixed) HMC for the phi^6 scalar field"};
    auto const cf      = app::common_flags(p, {.out = "phi6_f32.h5"});
    auto const& kappa  = p.opt<double>("kappa", 0.13, "hopping parameter");
    auto const& lambda = p.opt<double>("lambda", 0.05, "quartic coupling");
    auto const& g6     = p.opt<double>("g6", 0.01, "sextic coupling");
    auto const& ndim   = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const rf      = app::hmc_run_flags(p);
    if (!p.parse(argc, argv)) {
        return 0;
    }

    // ---- State: float lattice, action ----
    Lattice<float>::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(cf.L));
    Lattice<float> phi{shape};
    bool const resuming = rf.resuming();

    act::Phi6<float> phi6{.kappa  = static_cast<float>(kappa),
                          .lambda = static_cast<float>(lambda),
                          .g6     = static_cast<float>(g6)};

    // ---- Output: writer + series (all observables are double) ----
    io::Writer out            = app::open_writer(p, cf, argc, argv);
    std::string const outpath = app::out_path(cf);
    log::act(phi6);
    if (!resuming) {
        out.start_phase("therm");
    }
    out.start_phase("prod");
    auto s_therm  = resuming ? io::Series<double>{} : out.series<double>("/therm/stats/s");
    auto d_h      = out.series<double>("/prod/stats/dH");
    auto accepted = out.series<int>("/prod/stats/accepted");
    auto s_prod   = out.series<double>("/prod/obs/s");
    auto mag      = out.series<double>("/prod/obs/mag");
    auto m_sq     = out.series<double>("/prod/obs/m2");

    // ---- Updater (precision deduced from the float lattice + action) ----
    updater::Hmc hmc{phi6, phi, FastRng{cf.seed}, {.tau = rf.tau, .n_md = rf.n_md}};
    long long const start_i = app::resume_or_start(rf, phi, hmc, shape);

    // ---- Thermalisation ----
    if (!resuming) {
        log::info("hmc", "therm  {} trajectories", rf.n_therm);
        for (int i = 0; i < rf.n_therm; ++i) {
            (void)hmc.step(log::Mode::silent);
            s_therm.append(phi6.s_full(phi));
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
            s_prod.append(phi6.s_full(phi));
            mag.append(obs::mag_abs_of(s1, V));
            m_sq.append(obs::mean_of(s2, V));
        }
        if (rf.checkpoint_every > 0 && (i + 1) % rf.checkpoint_every == 0) {
            io::save_config(cfg_path(outpath, i + 1), phi, hmc.rng(), i + 1, argc, argv, &p);
        }
    }
}
