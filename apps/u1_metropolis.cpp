// Compact U(1) Wilson gauge theory — link Metropolis updater.
//
// Output schema:
//  /run@*                — reproducibility metadata stamped by Writer
//  /vars@*               — every --flag the Parser resolved
//  /therm/stats/s        — S per thermalisation sweep
//  /prod/stats/accept    — Metropolis accept rate per sweep
//  /prod/obs/s           — total Wilson action S per sweep
//  /prod/obs/plaq        — mean plaquette (S / (beta * n_plaq))

#include <reticolo/reticolo.hpp>

#include <cstddef>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Action = action::CompactU1<double>;

    // ---- CLI ----
    cli::Parser p{"u1_metropolis", "Compact U(1) Wilson action, link Metropolis"};
    auto const cf          = app::common_flags(p, {.L = 4, .out = "u1_metropolis.h5"});
    auto const& ndim       = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& beta       = p.opt<double>("beta", 1.0, "Wilson coupling");
    auto const& sigma      = p.opt<double>("sigma", 1.0, "Metropolis proposal stdev");
    auto const& n_therm    = p.opt<int>("n_therm", 200, "thermalisation sweeps");
    auto const& n_prod     = p.opt<int>("n_prod", 2000, "production sweeps");
    auto const& meas_every = p.opt<int>("meas_every", 1, "measure every N sweeps");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    io::Writer out = app::open_writer(p, cf, argc, argv);

    // ---- State: links, RNG, action ----
    LinkLattice<double>::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(cf.L));
    LinkLattice<double> links{shape, 0.0};
    FastRng rng{cf.seed};
    Action const action{.beta = beta};
    log::act(action);

    // ---- Output: series ----
    out.start_phase("therm");
    out.start_phase("prod");
    auto s_therm = out.series<double>("/therm/stats/s");
    auto accept  = out.series<double>("/prod/stats/accept");
    auto s_prod  = out.series<double>("/prod/obs/s");
    auto plaq    = out.series<double>("/prod/obs/plaq");

    // ---- Updater ----
    alg::Metropolis<Action, FastRng, double, LinkLattice<double>> metro{
        action, links, rng, alg::MetropolisSpec{.sigma = sigma}};

    // n_plaq = ndim*(ndim-1)/2 * V    where V = nsites
    std::size_t const v_sites = links.nsites();
    std::size_t const n_plaq =
        (static_cast<std::size_t>(ndim) * static_cast<std::size_t>(ndim - 1) / 2U) * v_sites;
    double const plaq_norm = (beta == 0.0) ? 1.0 : (beta * static_cast<double>(n_plaq));

    // ---- Thermalisation ----
    log::info("metr", "therm  {} sweeps", n_therm);
    for (int i = 0; i < n_therm; ++i) {
        (void)metro.step(log::Mode::silent);
        s_therm.append(action.s_full(links));
    }

    // ---- Production ----
    log::info("metr", "prod   {} sweeps", n_prod);
    for (int i = 0; i < n_prod; ++i) {
        auto const sweep_stats = metro.step();
        accept.append(sweep_stats.acceptance());
        if (i % meas_every == 0) {
            double const s = action.s_full(links);
            s_prod.append(s);
            // Paper convention: S = beta * sum cos(theta_p), so
            // mean plaquette <cos theta_p> = <S> / (beta * n_plaq).
            plaq.append(s / plaq_norm);
        }
    }
}
