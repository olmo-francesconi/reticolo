// Compact U(1) Wilson gauge theory — HMC updater (default Omelyan2).

#include <reticolo/reticolo.hpp>

#include <cstddef>
#include <filesystem>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Action = action::Wilson<math::group::U1, double>;

    // ---- CLI ----
    cli::Parser p{"u1_hmc", "Compact U(1) Wilson action, HMC (link-field)"};
    auto const& L          = p.opt<int>("L,size", 4, "linear lattice extent");
    auto const& ndim       = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& beta       = p.opt<double>("beta", 1.0, "Wilson coupling");
    auto const& tau        = p.opt<double>("tau", 1.0, "HMC trajectory length");
    auto const& n_md       = p.opt<int>("n_md", 20, "MD steps per trajectory");
    auto const& n_therm    = p.opt<int>("n_therm", 200, "thermalisation trajectories");
    auto const& n_prod     = p.opt<int>("n_prod", 2000, "production trajectories");
    auto const& meas_every = p.opt<int>("meas_every", 1, "measure every N trajectories");
    auto const& seed       = p.opt<unsigned long long>("seed", 42ULL, "RNG seed");
    auto const& workspace =
        p.opt<std::string>("workspace", std::string{"."}, "workspace folder (output + logs)");
    auto const& outfile = p.opt<std::string>(
        "out", std::string{"u1_hmc.h5"}, "HDF5 output file name, inside workspace");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    log::start(workspace, outfile);
    std::string const outpath = (std::filesystem::path{workspace} / outfile).string();

    // ---- State: links, RNG, action ----
    MatrixLinkLattice<math::group::U1, double>::SizeVec shape(static_cast<std::size_t>(ndim),
                                                              static_cast<std::size_t>(L));
    MatrixLinkLattice<math::group::U1, double> links{shape};
    FastRng rng{seed};
    Action const action{.beta = beta};
    log::act(action);

    // ---- Output: writer + series ----
    io::Writer out{outpath, argc, argv, &p};
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
