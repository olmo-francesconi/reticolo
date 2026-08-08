// Local Metropolis counterpart of `sine_gordon_hmc_f32` — same physics, sampled by a
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
    cli::Parser p{"sine_gordon_metropolis_f32",
                  "Single-precision (mixed) local Metropolis for the sine-Gordon field"};
    auto const cf     = app::common_flags(p, {.out = "sine_gordon_f32_metropolis.h5"});
    auto const& kappa = p.opt<double>("kappa", 1.0, "hopping parameter");
    auto const& alpha = p.opt<double>("alpha", 1.0, "cosine-potential strength");
    auto const& ndim  = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const rf     = app::metropolis_run_flags(p);
    if (!p.parse(argc, argv)) {
        return 0;
    }

    // ---- State: float lattice, action ----
    Lattice<float>::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(cf.L));
    Lattice<float> phi{shape};
    bool const resuming = rf.resuming();

    act::SineGordon<float> sg{.kappa = static_cast<float>(kappa),
                              .alpha = static_cast<float>(alpha)};

    // ---- Output: writer + series (all observables are double) ----
    io::Writer out            = app::open_writer(p, cf, argc, argv);
    std::string const outpath = app::out_path(cf);
    log::act(sg);
    if (!resuming) {
        out.start_phase("therm");
    }
    out.start_phase("prod");
    auto s_therm = resuming ? io::Series<double>{} : out.series<double>("/therm/stats/s");
    auto acc     = out.series<double>("/prod/stats/acceptance");
    auto s_prod  = out.series<double>("/prod/obs/s");
    auto mag     = out.series<double>("/prod/obs/mag");
    auto cos_phi = out.series<double>("/prod/obs/cos_phi");

    // ---- Updater (precision deduced from the float lattice + action) ----
    updater::Metropolis metro{sg, phi, FastRng{cf.seed}, {.sigma = rf.sigma}};
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
            auto const [s1, sc] = obs::reduce(
                phi, obs::kernel::phi, [](auto z) { return std::cos(static_cast<double>(z)); });
            s_prod.append(sg.s_full(phi));
            mag.append(obs::mag_abs_of(s1, V));
            cos_phi.append(obs::mean_of(sc, V));
        }
        if (rf.checkpoint_every > 0 && (i + 1) % rf.checkpoint_every == 0) {
            io::save_config(cfg_path(outpath, i + 1), phi, metro.rng(), i + 1, argc, argv, &p);
        }
    }
}
