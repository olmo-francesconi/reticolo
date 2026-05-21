// Metropolis for the O(3) sigma model on a `ndim`-dimensional cubic lattice.
//
//  S = -beta Σ_<x,y> phi(x) · phi(y),   phi(x) ∈ S^2
//
// N (= 3) is fixed at compile time because `OnSigma<N>` is templated; clone the
// app and change one number to study a different group.
//
// Output schema:
//  /run@*                — reproducibility metadata stamped by Writer
//  /vars@*               — every --flag the Parser resolved
//  /therm/stats/accept   — Metropolis acceptance per sweep
//  /prod/stats/accept    — Metropolis acceptance per sweep
//  /prod/obs/s           — S_full
//  /prod/obs/m2          — |M|^2 / V^2 (rotation-invariant)

#include <reticolo/reticolo.hpp>

#include <array>
#include <cstddef>

int main(int argc, char** argv) {
    using namespace reticolo;
    constexpr std::size_t k_n = 3;

    // ---- CLI ----
    cli::Parser p{"on_sigma_metropolis", "Metropolis for the O(3) sigma model"};
    auto const& L          = p.opt<int>("L,size", 8, "linear lattice extent");
    auto const& beta       = p.opt<double>("beta", 0.7, "inverse temperature");
    auto const& ndim       = p.opt<int>("ndim", 3, "spatial dimensions (2 or 3)");
    auto const& n_therm    = p.opt<int>("n_therm", 400, "thermalisation sweeps");
    auto const& n_prod     = p.opt<int>("n_prod", 2000, "production sweeps");
    auto const& meas_every = p.opt<int>("meas_every", 1, "measure every N sweeps");
    auto const& seed       = p.opt<unsigned long long>("seed", 42ULL, "RNG seed");
    auto const& outpath = p.opt<std::string>("out", std::string{"on_sigma.h5"}, "HDF5 output path");
    if (!p.parse(argc, argv))
        return 0;

    log::start(outpath);

    // ---- State: lattice, RNG, action ----
    Lattice<std::array<double, k_n>>::SizeVec shape(static_cast<std::size_t>(ndim),
                                                    static_cast<std::size_t>(L));
    Lattice<std::array<double, k_n>> phi{shape};
    FastRng rng{seed};
    act::OnSigma<k_n> on{.beta = beta};
    log::act(on);

    // Cold start: every spin aligned along the first axis. The thermalisation
    // sweeps below decorrelate from this configuration before measurement.
    for (Site const x : phi.sites()) {
        phi[x] = {1.0, 0.0, 0.0};
    }

    // ---- Output: writer + series ----
    io::Writer out{outpath, argc, argv, &p};
    out.start_phase("therm");
    out.start_phase("prod");
    auto accept_therm = out.series<double>("/therm/stats/accept");
    auto accept_prod  = out.series<double>("/prod/stats/accept");
    auto s_prod       = out.series<double>("/prod/obs/s");
    auto m2_prod      = out.series<double>("/prod/obs/m2");

    // ---- Updater ----
    alg::Metropolis<act::OnSigma<k_n>, FastRng> mc{on, phi, rng, alg::MetropolisSpec{.sigma = 0.0}};

    // ---- Thermalisation ----
    log::info("metr", "therm  {} sweeps", n_therm);
    for (int i = 0; i < n_therm; ++i) {
        accept_therm.append(mc.step(log::Mode::silent).acceptance());
    }

    // ---- Production ----
    log::info("metr", "prod   {} sweeps", n_prod);
    for (int i = 0; i < n_prod; ++i) {
        accept_prod.append(mc.step().acceptance());
        if (i % meas_every == 0) {
            s_prod.append(on.s_full(phi));
            m2_prod.append(obs::mag::on_sq<k_n>(phi));
        }
    }
}
