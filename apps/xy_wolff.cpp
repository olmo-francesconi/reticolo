// Wolff cluster + Metropolis hybrid for the XY (planar rotor) model on a 2D
// square lattice.
//
//  S = -beta Σ_<x,y> cos(theta(x) - theta(y))
//
// Each measurement step does `n_cluster` single-cluster Wolff updates followed
// by one full Metropolis sweep (the sweep covers the few sites that Wolff
// rarely touches deep in the symmetric phase).
//
// Output schema:
//  /run@*                — reproducibility metadata stamped by Writer
//  /vars@*               — every --flag the Parser resolved
//  /therm/stats/cluster  — cluster size per Wolff update
//  /prod/stats/cluster   — cluster size per Wolff update
//  /prod/stats/accept    — Metropolis acceptance per sweep
//  /prod/obs/s           — S_full
//  /prod/obs/m2          — |M|^2 / V^2     (rotation-invariant magnetization²)

#include <reticolo/reticolo.hpp>

#include <cstddef>
#include <numbers>

int main(int argc, char** argv) {
    using namespace reticolo;

    cli::Parser p{"xy_wolff", "Wolff cluster + Metropolis hybrid for the XY model"};
    auto const& L          = p.opt<int>("L,size", 16, "linear lattice extent (2D)");
    auto const& beta       = p.opt<double>("beta", 1.12, "inverse temperature");
    auto const& sigma      = p.opt<double>("sigma", 0.5, "Metropolis Gaussian step width");
    auto const& n_cluster  = p.opt<int>("n_cluster", 4, "Wolff updates per measurement");
    auto const& n_therm    = p.opt<int>("n_therm", 200, "thermalisation measurements");
    auto const& n_prod     = p.opt<int>("n_prod", 2000, "production measurements");
    auto const& meas_every = p.opt<int>("meas_every", 1, "measure every N measurements");
    auto const& seed       = p.opt<unsigned long long>("seed", 42ULL, "RNG seed");
    auto const& outpath = p.opt<std::string>("out", std::string{"xy_wolff.h5"}, "HDF5 output path");
    if (!p.parse(argc, argv))
        return 0;

    log::start(outpath);

    auto const l_sz = static_cast<std::size_t>(L);
    Lattice<double> theta{{l_sz, l_sz}};
    FastRng rng{seed};
    for (Site const x : theta.sites()) {
        theta[x] = 2.0 * std::numbers::pi * rng.uniform();
    }
    act::Xy<double> xy{.beta = beta};
    log::act(xy);

    io::Writer out{outpath, argc, argv, &p};
    out.start_phase("therm");
    out.start_phase("prod");
    auto cluster_therm = out.series<std::size_t>("/therm/stats/cluster");
    auto cluster_prod  = out.series<std::size_t>("/prod/stats/cluster");
    auto accept_prod   = out.series<double>("/prod/stats/accept");
    auto s_prod        = out.series<double>("/prod/obs/s");
    auto m2_prod       = out.series<double>("/prod/obs/m2");

    alg::Wolff<act::Xy<double>, FastRng> wolff{xy, theta, rng};
    alg::Metropolis<act::Xy<double>, FastRng> mc{
        xy, theta, rng, alg::MetropolisSpec{.sigma = sigma}};

    log::info("wolf", "therm  {} measurements × {} clusters + 1 sweep", n_therm, n_cluster);
    for (int i = 0; i < n_therm; ++i) {
        for (int k = 0; k < n_cluster; ++k) {
            cluster_therm.append(wolff.step(log::Mode::silent).cluster_size);
        }
        (void)mc.step(log::Mode::silent);
    }
    log::info("wolf", "prod   {} measurements × {} clusters + 1 sweep", n_prod, n_cluster);
    for (int i = 0; i < n_prod; ++i) {
        for (int k = 0; k < n_cluster; ++k) {
            cluster_prod.append(wolff.step().cluster_size);
        }
        auto const stats = mc.step();
        accept_prod.append(stats.acceptance());
        if (i % meas_every == 0) {
            s_prod.append(xy.s_full(theta));
            m2_prod.append(obs::mag::xy_sq(theta));
        }
    }
}
