// Phase-quenched local Metropolis for the Bose gas — the same physics as
// `bose_gas_hmc`, sampled by a checkerboard sweep instead of a trajectory.
//
// The sweep samples exp(-S_R) exactly as HMC does. That the local-energy identity
// survives the anisotropy is not an accident: the hopping term is
// -2*Re(conj(phi_x)*weighted_nbr) and Re(conj(a)*b) is symmetric, so a bond
// contributes equally to either endpoint. The chemical potential's
// forward/backward asymmetry -- the source of the sign problem -- lives entirely
// in S_I, which the sampled weight never touches.
//
// Every extent must be EVEN (bipartite site parity under periodic wrap).
//
// Output schema:
//  /run@*                 -- reproducibility metadata stamped by Writer
//  /vars@*                -- every --flag the Parser resolved
//  /therm/stats/s         -- S_R per thermalisation sweep
//  /prod/stats/acceptance -- accepted fraction of the sweep's V proposals
//  /prod/obs/s_r          -- S_R (carried incrementally by the updater)
//  /prod/obs/s_i          -- S_I (the reweighting phase; measured, not sampled)

#include <reticolo/reticolo.hpp>

#include <array>
#include <complex>
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
    using Action = act::BoseGas<double>;

    // ---- CLI ----
    cli::Parser p{"bose_gas_metropolis",
                  "Phase-quenched local Metropolis for the 4D Bose gas (records S_R + S_I)"};
    auto const cf      = app::common_flags(p, {.L = 4, .out = "bose_gas_metropolis.h5"});
    auto const& ndim   = p.opt<int>("ndim", 4, "spacetime dimensions");
    auto const& mass   = p.opt<double>("mass", 1.0, "bare mass m");
    auto const& lambda = p.opt<double>("lambda", 1.0, "quartic coupling lambda");
    auto const& mu     = p.opt<double>("mu", 1.0, "chemical potential mu");
    auto const rf = app::metropolis_run_flags(p, {.sigma = 0.3, .n_therm = 500, .n_prod = 5000});
    if (!p.parse(argc, argv)) {
        return 0;
    }

    // ---- State: lattice, action ----
    Lattice<std::complex<double>>::SizeVec shape(static_cast<std::size_t>(ndim),
                                                 static_cast<std::size_t>(cf.L));
    Lattice<std::complex<double>> phi{shape};
    bool const resuming = rf.resuming();

    Action const action{.mass = mass, .lambda = lambda, .mu = mu};

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
    auto s_r     = out.series<double>("/prod/obs/s_r");
    auto s_i     = out.series<double>("/prod/obs/s_i");

    // ---- Updater ----
    // The proposal is complex: fill_gaussian gives independent N(0,1) on Re and
    // Im, so one move explores both components rather than needing two sweeps.
    updater::Metropolis metro{action, phi, FastRng{cf.seed}, {.sigma = rf.sigma}};
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
            // S_R is carried incrementally; S_I is the sign-problem observable and
            // is NOT part of the sampled weight, so it still needs its own sweep.
            s_r.append(metro.last_s_full());
            s_i.append(action.s_imag(phi));
        }
        if (rf.checkpoint_every > 0 && (i + 1) % rf.checkpoint_every == 0) {
            io::save_config(cfg_path(outpath, i + 1), phi, metro.rng(), i + 1, argc, argv, &p);
        }
    }
}
