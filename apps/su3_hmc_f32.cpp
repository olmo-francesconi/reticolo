// SU(3) Wilson gauge theory — single-precision (mixed) HMC. Mixed-precision
// counterpart of su3_hmc: links, momenta and the staple force run in `float`
// (the force is fully 4-/8-wide via the batched kernel), while the action
// reduction, kinetic energy and ΔH accumulate in `double`, and the
// Cayley-Hamilton group exponential is evaluated in double on float storage.
// Precision derives from MatrixLinkLattice<SU3, T> — swap float for double for
// full double precision with no other change.

#include <reticolo/reticolo.hpp>

#include <cstddef>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Group  = gauge_group::SU3;
    using Action = action::Wilson<Group, float>;
    using Field  = MatrixLinkLattice<Group, float>;

    cli::Parser p{"su3_hmc_f32", "SU(3) Wilson action, single-precision (mixed) HMC"};
    auto const cf          = app::common_flags(p, {.L = 4, .out = "su3_hmc_f32.h5"});
    auto const& ndim       = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& beta       = p.opt<double>("beta", 5.7, "Wilson coupling");
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

    // ---- State: links (cold-started to identity), RNG, action ----
    Field::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(cf.L));
    Field links{shape};
    std::size_t const ns = links.nsites();
    for (std::size_t mu = 0; mu < static_cast<std::size_t>(ndim); ++mu) {
        float* const blk = links.mu_block_data(mu);
        for (std::size_t s = 0; s < ns; ++s) {
            blk[(0 * ns) + s]  = 1.0F;  // Re U_00
            blk[(8 * ns) + s]  = 1.0F;  // Re U_11
            blk[(16 * ns) + s] = 1.0F;  // Re U_22
        }
    }
    FastRng rng{cf.seed};
    Action const action{.beta = static_cast<float>(beta)};
    log::act(action);

    out.start_phase("therm");
    out.start_phase("prod");
    auto s_therm  = out.series<double>("/therm/stats/s");
    auto d_h      = out.series<double>("/prod/stats/dH");
    auto accepted = out.series<int>("/prod/stats/accepted");
    auto s_prod   = out.series<double>("/prod/obs/s");
    auto plaq     = out.series<double>("/prod/obs/plaq");

    alg::Hmc hmc{action, links, rng, {.tau = tau, .n_md = n_md}, alg::integ::omelyan2};

    std::size_t const n_plaq =
        (static_cast<std::size_t>(ndim) * static_cast<std::size_t>(ndim - 1) / 2U) * ns;
    double const plaq_norm = (beta == 0.0) ? 1.0 : (beta * static_cast<double>(n_plaq));

    log::info("hmc", "therm  {} trajectories", n_therm);
    for (int i = 0; i < n_therm; ++i) {
        (void)hmc.step(log::Mode::silent);
        s_therm.append(action.s_full(links));
    }

    log::info("hmc", "prod   {} trajectories", n_prod);
    for (int i = 0; i < n_prod; ++i) {
        auto const step = hmc.step();
        d_h.append(step.dH);
        accepted.append(step.accepted ? 1 : 0);
        if (i % meas_every == 0) {
            double const s = action.s_full(links);
            s_prod.append(s);
            plaq.append(1.0 - (s / plaq_norm));
        }
    }
}
