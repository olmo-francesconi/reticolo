// SU(3) Wilson gauge action sampled by the local Metropolis updater — the same
// physics as `su3_hmc`, with a link sweep in place of a trajectory. A link
// sweep is 2*ndim colour passes, one per (direction, site parity): every link in
// the staple of (x,mu) is either of another direction or on the opposite parity,
// so each pass writes links nothing in that pass reads.
//
// The proposal is U -> exp(sigma*H)*U with H the SAME algebra element HMC draws
// for its momenta; --sigma tunes the acceptance (SU(3) wants a smaller sigma than
// SU(2) at comparable acceptance — more generators per link).
//
// Every extent must be EVEN (bipartite site parity under periodic wrap).
//
// Output schema:
//  /run@*                 -- reproducibility metadata stamped by Writer
//  /vars@*                -- every --flag the Parser resolved
//  /therm/stats/s         -- S_full per thermalisation sweep
//  /prod/stats/acceptance -- accepted fraction over the sweep's links
//  /prod/obs/s            -- S_full (carried incrementally by the updater)
//  /prod/obs/plaq         -- <Re Tr U_p>/N

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
    using Group  = math::group::SU3;
    using Action = action::Wilson<Group, double>;
    using Field  = MatrixLinkLattice<Group, double>;

    // ---- CLI ----
    cli::Parser p{"su3_metropolis", "SU(3) Wilson action, local Metropolis (matrix-link field)"};
    auto const cf    = app::common_flags(p, {.L = 4, .out = "su3_metropolis.h5"});
    auto const& ndim = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& beta = p.opt<double>("beta", 6.0, "Wilson coupling");
    auto const rf    = app::metropolis_run_flags(p, {.sigma = 0.3, .n_prod = 2000});
    if (!p.parse(argc, argv)) {
        return 0;
    }

    // ---- State: links (cold-started to identity unless resuming) ----
    Field::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(cf.L));
    Field links{shape};
    bool const resuming  = rf.resuming();
    std::size_t const ns = links.nsites();
    if (!resuming) {
        // Cold start: every link = 3×3 identity (Re U_{ii} = 1, all other slots 0).
        set_cold_identity(links);
    }
    Action const action{.beta = beta};

    // ---- Output: writer + series ----
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
    // A link sweep is 2*ndim colour passes — one per (direction, site parity).
    updater::Metropolis metro{action, links, FastRng{cf.seed}, {.sigma = rf.sigma}};
    long long const start_i = app::resume_or_start(rf, links, metro, shape);
    if (resuming) {
        metro.resync_s_full();  // the carried S belongs to the pre-resume field
    }

    std::size_t const n_plaq =
        (static_cast<std::size_t>(ndim) * static_cast<std::size_t>(ndim - 1) / 2U) * ns;
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
            // Carried incrementally by the updater — no extra O(V) sweep.
            double const s = metro.last_s_full();
            s_prod.append(s);
            plaq.append(1.0 - (s / plaq_norm));
        }
        if (rf.checkpoint_every > 0 && (i + 1) % rf.checkpoint_every == 0) {
            io::save_config(cfg_path(outpath, i + 1), links, metro.rng(), i + 1, argc, argv, &p);
        }
    }
}
