// Compact U(1) Wilson gauge theory — HMC updater (default Omelyan2).

#include <reticolo/reticolo.hpp>

#include <cstddef>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Action = action::CompactU1<double>;

    // ---- CLI ----
    cli::Parser p{"u1_hmc", "Compact U(1) Wilson action, HMC (link-field)"};
    auto const cf          = app::common_flags(p, {.L = 4, .out = "u1_hmc.h5"});
    auto const& ndim       = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& beta       = p.opt<double>("beta", 1.0, "Wilson coupling");
    auto const& tau        = p.opt<double>("tau", 1.0, "HMC trajectory length");
    auto const& n_md       = p.opt<int>("n_md", 20, "MD steps per trajectory");
    auto const& n_therm    = p.opt<int>("n_therm", 200, "thermalisation trajectories");
    auto const& n_prod     = p.opt<int>("n_prod", 2000, "production trajectories");
    auto const& meas_every = p.opt<int>("meas_every", 1, "measure every N trajectories");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    // ---- Output: writer ----
    io::Writer out = app::open_writer(p, cf, argc, argv);

    // ---- State: links, RNG, action ----
    LinkLattice<double>::SizeVec shape(static_cast<std::size_t>(ndim),
                                       static_cast<std::size_t>(cf.L));
    LinkLattice<double> links{shape, 0.0};
    FastRng rng{cf.seed};
    Action const action{.beta = beta};
    log::act(action);

    // ---- Output: series ----
    out.start_phase("therm");
    out.start_phase("prod");
    auto s_therm  = out.series<double>("/therm/stats/s");
    auto d_h      = out.series<double>("/prod/stats/dH");
    auto accepted = out.series<int>("/prod/stats/accepted");
    auto s_prod   = out.series<double>("/prod/obs/s");
    auto plaq     = out.series<double>("/prod/obs/plaq");

    // ---- Updater ----
    alg::Hmc hmc{action, links, rng, {.tau = tau, .n_md = n_md}, alg::integ::omelyan2};

    std::size_t const v_sites = links.nsites();
    std::size_t const n_plaq =
        (static_cast<std::size_t>(ndim) * static_cast<std::size_t>(ndim - 1) / 2U) * v_sites;
    double const plaq_norm = (beta == 0.0) ? 1.0 : (beta * static_cast<double>(n_plaq));

    // ---- Thermalisation ----
    log::info("hmc", "therm  {} trajectories", n_therm);
    for (int i = 0; i < n_therm; ++i) {
        (void)hmc.step(log::Mode::silent);
        s_therm.append(action.s_full(links));
    }

    // ---- Production ----
    log::info("hmc", "prod   {} trajectories", n_prod);
    for (int i = 0; i < n_prod; ++i) {
        auto const step = hmc.step();
        d_h.append(step.dH);
        accepted.append(step.accepted ? 1 : 0);
        if (i % meas_every == 0) {
            double const s = action.s_full(links);
            s_prod.append(s);
            // Paper convention: S = beta * sum cos(theta_p), so
            // mean plaquette <cos theta_p> = <S> / (beta * n_plaq).
            plaq.append(s / plaq_norm);
        }
    }
}
