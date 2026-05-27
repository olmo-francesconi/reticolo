// SU(3) Wilson gauge theory — HMC updater (default Omelyan2).
//
// Storage: MatrixLinkLattice<SU3, double>, 18 reals per link (full 3×3
// complex matrix). Wilson action S_W = (β/3)·Σ_p (3 − Re Tr U_p), bounded
// below by 0 at the cold start. Mean plaquette ⟨P⟩ = (1/3)⟨Re Tr U_p⟩.
// Weak-coupling reference: ⟨P⟩ ≈ 1 − 1/(3β).

#include <reticolo/reticolo.hpp>

#include <cstddef>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Group  = gauge_group::SU3;
    using Action = action::Wilson<Group, double>;
    using Field  = MatrixLinkLattice<Group, double>;

    // ---- CLI ----
    cli::Parser p{"su3_hmc", "SU(3) Wilson action, HMC (matrix-link field)"};
    auto const& L          = p.opt<int>("L,size", 4, "linear lattice extent");
    auto const& ndim       = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& beta       = p.opt<double>("beta", 6.0, "Wilson coupling");
    auto const& tau        = p.opt<double>("tau", 1.0, "HMC trajectory length");
    auto const& n_md       = p.opt<int>("n_md", 30, "MD steps per trajectory");
    auto const& n_therm    = p.opt<int>("n_therm", 200, "thermalisation trajectories");
    auto const& n_prod     = p.opt<int>("n_prod", 2000, "production trajectories");
    auto const& meas_every = p.opt<int>("meas_every", 1, "measure every N trajectories");
    auto const& seed       = p.opt<unsigned long long>("seed", 42ULL, "RNG seed");
    auto const& outpath = p.opt<std::string>("out", std::string{"su3_hmc.h5"}, "HDF5 output path");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    log::start(outpath);

    // ---- State: links (cold-started to identity), RNG, action ----
    Field::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(L));
    Field links{shape};
    // Cold start: every link = 3×3 identity (Re U_{ii} = 1, all other slots 0).
    std::size_t const ns = links.nsites();
    for (std::size_t mu = 0; mu < static_cast<std::size_t>(ndim); ++mu) {
        double* const blk = links.mu_block_data(mu);
        for (std::size_t s = 0; s < ns; ++s) {
            blk[(0 * ns) + s]  = 1.0;  // Re U_{00}
            blk[(8 * ns) + s]  = 1.0;  // Re U_{11}
            blk[(16 * ns) + s] = 1.0;  // Re U_{22}
        }
    }
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

    std::size_t const n_plaq =
        (static_cast<std::size_t>(ndim) * static_cast<std::size_t>(ndim - 1) / 2U) * ns;
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
            plaq.append(1.0 - (s / plaq_norm));
        }
    }
}
