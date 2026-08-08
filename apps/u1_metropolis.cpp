// Local Metropolis counterpart of `u1_hmc` — same physics, sampled by a
// checkerboard sweep instead of a trajectory. The loop unit is a SWEEP (one
// lattice pass), so the counts are NOT comparable to the HMC app at the same
// number. Every extent must be EVEN (bipartite parity under periodic wrap).
//
// Output schema matches the HMC app except /prod/stats: a sweep has no single
// accept decision, so it records /prod/stats/acceptance in place of dH+accepted.

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
    using Action = action::Wilson<math::group::U1, double>;
    using Field  = MatrixLinkLattice<math::group::U1, double>;

    // ---- CLI ----
    cli::Parser p{"u1_metropolis", "Compact U(1) Wilson action, local Metropolis (link-field)"};
    auto const cf    = app::common_flags(p, {.L = 4, .out = "u1_hmc.h5"});
    auto const& ndim = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& beta = p.opt<double>("beta", 1.0, "Wilson coupling");
    auto const rf    = app::metropolis_run_flags(p, {.sigma = 0.3});
    if (!p.parse(argc, argv)) {
        return 0;
    }

    // ---- State: links, action ----
    Field::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(cf.L));
    Field links{shape};
    bool const resuming = rf.resuming();
    Action const action{.beta = beta};

    // ---- Output: writer ----
    io::Writer out            = app::open_writer(p, cf, argc, argv);
    std::string const outpath = app::out_path(cf);
    log::act(action);
    if (!resuming) {
        out.start_phase("therm");
    }
    out.start_phase("prod");
    auto s_therm = resuming ? io::Series<double>{} : out.series<double>("/therm/stats/s");
    auto acc     = out.series<double>("/prod/stats/acceptance");
    auto s_prod  = out.series<double>("/prod/obs/s");
    auto plaq    = out.series<double>("/prod/obs/plaq");

    // ---- Updater ----
    updater::Metropolis metro{action, links, FastRng{cf.seed}, {.sigma = rf.sigma}};
    long long const start_i = app::resume_or_start(rf, links, metro, shape);
    if (resuming) {
        metro.resync_s_full();  // the carried S belongs to the pre-resume field
    }

    std::size_t const v_sites = links.nsites();
    std::size_t const n_plaq =
        (static_cast<std::size_t>(ndim) * static_cast<std::size_t>(ndim - 1) / 2U) * v_sites;
    double const plaq_norm = (beta == 0.0) ? 1.0 : (beta * static_cast<double>(n_plaq));

    // ---- Thermalisation ----
    if (!resuming) {
        log::info("metr", "therm  {} sweeps", rf.n_therm);
        for (int i = 0; i < rf.n_therm; ++i) {
            (void)metro.step(log::Mode::silent);
            s_therm.append(metro.last_s_full());
        }
    }

    // ---- Production ----
    log::info("metr", "prod   {} sweeps (from {})", rf.n_prod, start_i);
    for (long long i = start_i; i < rf.n_prod; ++i) {
        auto const step = metro.step();
        acc.append(step.acceptance());
        if (i % rf.meas_every == 0) {
            double const s = action.s_full(links);
            s_prod.append(s);
            // Paper convention: S = beta * sum cos(theta_p), so
            // mean plaquette <cos theta_p> = <S> / (beta * n_plaq).
            plaq.append(s / plaq_norm);
        }
        if (rf.checkpoint_every > 0 && (i + 1) % rf.checkpoint_every == 0) {
            io::save_config(cfg_path(outpath, i + 1), links, metro.rng(), i + 1, argc, argv, &p);
        }
    }
}
