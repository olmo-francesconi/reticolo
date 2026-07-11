// Compact U(1) Wilson gauge theory — single-precision (mixed) HMC. The
// mixed-precision counterpart of u1_hmc: the link field, momentum, staple force
// and the group exponential run in `float`, while the action reduction S, the
// kinetic energy and ΔH accumulate in `double` so the Metropolis acceptance
// still targets exp(-S). Precision is derived from the
// MatrixLinkLattice<U1, T> type — swap `float` for `double` for full double.

#include <reticolo/reticolo.hpp>

#include <cstddef>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Action = action::Wilson<math::group::U1, float>;

    // ---- CLI ----
    cli::Parser p{"u1_hmc_f32", "Compact U(1) Wilson action, single-precision (mixed) HMC"};
    auto const cf          = app::common_flags(p, {.L = 4, .out = "u1_hmc_f32.h5"});
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

    // ---- State: float links, RNG, float action ----
    MatrixLinkLattice<math::group::U1, float>::SizeVec shape(static_cast<std::size_t>(ndim),
                                                             static_cast<std::size_t>(cf.L));
    MatrixLinkLattice<math::group::U1, float> links{shape};
    Action const action{.beta = static_cast<float>(beta)};
    log::act(action);

    // ---- Output: series (all observables are double) ----
    out.start_phase("therm");
    out.start_phase("prod");
    auto s_therm  = out.series<double>("/therm/stats/s");
    auto d_h      = out.series<double>("/prod/stats/dH");
    auto accepted = out.series<int>("/prod/stats/accepted");
    auto s_prod   = out.series<double>("/prod/obs/s");
    auto plaq     = out.series<double>("/prod/obs/plaq");

    // ---- Updater (precision deduced from the float link field + action) ----
    updater::Hmc hmc{action, links, FastRng{cf.seed}, {.tau = tau, .n_md = n_md}};

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
            plaq.append(s / plaq_norm);
        }
    }
}
