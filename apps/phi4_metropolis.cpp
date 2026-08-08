// Local Metropolis for the phi^4 scalar field on a `ndim`-dimensional hypercubic
// lattice — the same physics as `phi4_hmc`, sampled by the checkerboard local
// updater instead of a trajectory. The unit of the loop is a SWEEP (one pass over
// every site), so --n_therm / --n_prod are NOT comparable to the HMC app's counts
// at the same number: an HMC trajectory is n_md lattice passes.
//
// Every extent must be EVEN — the checkerboard needs a bipartite lattice under
// periodic wrap; the updater refuses an odd one at construction.
//
// Output schema:
//  /run@*                 — reproducibility metadata stamped by Writer
//  /vars@*                — every --flag the Parser resolved
//  /therm/stats/s         — S_full per thermalisation sweep
//  /prod/stats/acceptance — accepted fraction of the sweep's V proposals
//  /prod/obs/s            — S_full (carried incrementally by the updater)
//  /prod/obs/mag          — |<phi>|           (per-config magnetization magnitude)
//  /prod/obs/mag_sq       — (<phi>)^2         (per-config magnetization squared; chi input)
//  /prod/obs/m2           — <phi^2>           (per-site field-squared average)

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
    cli::Parser p{"phi4_metropolis", "Local Metropolis for the phi^4 scalar field"};
    auto const cf      = app::common_flags(p, {.out = "phi4_metropolis.h5"});
    auto const& kappa  = p.opt<double>("kappa", 0.18, "hopping parameter");
    auto const& lambda = p.opt<double>("lambda", 1.0, "quartic coupling");
    auto const& ndim   = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const rf      = app::metropolis_run_flags(p);
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
    auto s_therm = resuming ? io::Series<double>{} : out.series<double>("/therm/stats/s");
    auto acc     = out.series<double>("/prod/stats/acceptance");
    auto s_prod  = out.series<double>("/prod/obs/s");
    auto mag     = out.series<double>("/prod/obs/mag");
    auto mag_sq  = out.series<double>("/prod/obs/mag_sq");
    auto m_sq    = out.series<double>("/prod/obs/m2");

    // ---- Updater: owns the RNG (one site stream per canonical slab) ----
    updater::Metropolis metro{phi4, phi, FastRng{cf.seed}, {.sigma = rf.sigma}};
    long long const start_i = app::resume_or_start(rf, phi, metro, shape);
    if (resuming) {
        metro.resync_s_full();  // the carried S belongs to the pre-resume field
    }

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
            auto const V        = static_cast<double>(phi.nsites());
            auto const [s1, s2] = obs::reduce(phi, obs::kernel::phi, obs::kernel::phi_sq);
            // The updater carries S incrementally, so this costs no extra sweep.
            s_prod.append(metro.last_s_full());
            mag.append(obs::mag_abs_of(s1, V));
            mag_sq.append(obs::sq_of_mean_of(s1, V));
            m_sq.append(obs::mean_of(s2, V));
        }
        if (rf.checkpoint_every > 0 && (i + 1) % rf.checkpoint_every == 0) {
            io::save_config(cfg_path(outpath, i + 1), phi, metro.rng(), i + 1, argc, argv, &p);
        }
    }
}
