// SU(3) Wilson gauge theory — single-precision (mixed) HMC. Mixed-precision
// counterpart of su3_hmc: links, momenta and the staple force run in `float`
// (the force is fully 4-/8-wide via the batched kernel), while the action
// reduction, kinetic energy and ΔH accumulate in `double`, and the
// Cayley-Hamilton group exponential is evaluated in double on float storage.
// Precision derives from MatrixLinkLattice<SU3, T> — swap float for double for
// full double precision with no other change.

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
    using Action = action::Wilson<Group, float>;
    using Field  = MatrixLinkLattice<Group, float>;

    cli::Parser p{"su3_hmc_f32", "SU(3) Wilson action, single-precision (mixed) HMC"};
    auto const cf    = app::common_flags(p, {.L = 4, .out = "su3_hmc_f32.h5"});
    auto const& ndim = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& beta = p.opt<double>("beta", 5.7, "Wilson coupling");
    auto const rf    = app::hmc_run_flags(p, {.n_prod = 2000});
    if (!p.parse(argc, argv)) {
        return 0;
    }

    // ---- State: links (cold-started to identity unless resuming) ----
    Field::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(cf.L));
    Field links{shape};
    bool const resuming  = rf.resuming();
    std::size_t const ns = links.nsites();
    if (!resuming) {
        set_cold_identity(links);
    }
    Action const action{.beta = static_cast<float>(beta)};

    io::Writer out            = app::open_writer(p, cf, argc, argv);
    std::string const outpath = app::out_path(cf);
    log::act(action);
    if (!resuming) {
        out.start_phase("therm");
    }
    out.start_phase("prod");
    auto s_therm  = resuming ? io::Series<double>{} : out.series<double>("/therm/stats/s");
    auto d_h      = out.series<double>("/prod/stats/dH");
    auto accepted = out.series<int>("/prod/stats/accepted");
    auto s_prod   = out.series<double>("/prod/obs/s");
    auto plaq     = out.series<double>("/prod/obs/plaq");

    updater::Hmc hmc{action, links, FastRng{cf.seed}, {.tau = rf.tau, .n_md = rf.n_md}};
    long long const start_i = app::resume_or_start(rf, links, hmc, shape);

    std::size_t const n_plaq =
        (static_cast<std::size_t>(ndim) * static_cast<std::size_t>(ndim - 1) / 2U) * ns;
    double const plaq_norm = (beta == 0.0) ? 1.0 : (beta * static_cast<double>(n_plaq));

    if (!resuming) {
        log::info("hmc", "therm  {} trajectories", rf.n_therm);
        for (int i = 0; i < rf.n_therm; ++i) {
            (void)hmc.step(log::Mode::silent);
            s_therm.append(action.s_full(links));
        }
    }

    log::info("hmc", "prod   {} trajectories (from {})", rf.n_prod, start_i);
    for (long long i = start_i; i < rf.n_prod; ++i) {
        auto const step = hmc.step();
        d_h.append(step.dH);
        accepted.append(step.accepted ? 1 : 0);
        if (i % rf.meas_every == 0) {
            double const s = action.s_full(links);
            s_prod.append(s);
            plaq.append(1.0 - (s / plaq_norm));
        }
        if (rf.checkpoint_every > 0 && (i + 1) % rf.checkpoint_every == 0) {
            io::save_config(cfg_path(outpath, i + 1), links, hmc.rng(), i + 1, argc, argv, &p);
        }
    }
}
