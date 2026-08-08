// Local Metropolis counterpart of `xy_hmc` — same physics, sampled by a
// checkerboard sweep instead of a trajectory. The loop unit is a SWEEP (one
// lattice pass), so the counts are NOT comparable to the HMC app at the same
// number. Every extent must be EVEN (bipartite parity under periodic wrap).
//
// Output schema matches the HMC app except /prod/stats: a sweep has no single
// accept decision, so it records /prod/stats/acceptance in place of dH+accepted.

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
    cli::Parser p{"xy_metropolis", "Local Metropolis for the XY (planar rotor) model"};
    auto const cf    = app::common_flags(p, {.out = "xy_metropolis.h5"});
    auto const& beta = p.opt<double>("beta", 0.45, "inverse-temperature coupling");
    auto const& ndim = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const rf    = app::metropolis_run_flags(p);
    if (!p.parse(argc, argv)) {
        return 0;
    }

    // ---- State: lattice, action ----
    Lattice<double>::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(cf.L));
    Lattice<double> theta{shape};
    bool const resuming = rf.resuming();

    act::Xy<double> xy{.beta = beta};

    // ---- Output: writer + series ----
    io::Writer out            = app::open_writer(p, cf, argc, argv);
    std::string const outpath = app::out_path(cf);
    log::act(xy);
    if (!resuming) {
        out.start_phase("therm");
    }
    out.start_phase("prod");
    auto s_therm = resuming ? io::Series<double>{} : out.series<double>("/therm/stats/s");
    auto acc     = out.series<double>("/prod/stats/acceptance");
    auto s_prod  = out.series<double>("/prod/obs/s");
    auto mag     = out.series<double>("/prod/obs/mag");

    // ---- Updater ----
    updater::Metropolis metro{xy, theta, FastRng{cf.seed}, {.sigma = rf.sigma}};
    long long const start_i = app::resume_or_start(rf, theta, metro, shape);
    if (resuming) {
        metro.resync_s_full();  // the carried S belongs to the pre-resume field
    }

    // |<(cos θ, sin θ)>| — the XY order parameter (0 disordered, 1 aligned).
    // Two lanes fused into one sweep, like every other observable in the tree.
    auto xy_mag = [&theta]() {
        auto const [cx, cy] = obs::reduce(
            theta, [](double t) { return std::cos(t); }, [](double t) { return std::sin(t); });
        double const inv = 1.0 / static_cast<double>(theta.nsites());
        return std::hypot(cx * inv, cy * inv);
    };

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
            s_prod.append(xy.s_full(theta));
            mag.append(xy_mag());
        }
        if (rf.checkpoint_every > 0 && (i + 1) % rf.checkpoint_every == 0) {
            io::save_config(cfg_path(outpath, i + 1), theta, metro.rng(), i + 1, argc, argv, &p);
        }
    }
}
