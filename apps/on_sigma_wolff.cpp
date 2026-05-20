// Single-cluster Wolff (+ Metropolis safety sweep) for the O(3) sigma model
// on a `ndim`-dimensional cubic lattice.
//
//   S = -beta Σ_<x,y> phi(x) · phi(y),   phi(x) ∈ S^2
//
// At criticality cluster updates beat single-spin Metropolis by orders of
// magnitude in autocorrelation time — use this app whenever the science
// question lives near beta_c.
//
// Output schema:
//   /run@*                — reproducibility metadata stamped by Writer
//   /vars@*               — every --flag the Parser resolved
//   /therm/stats/cluster  — cluster size per Wolff update
//   /prod/stats/cluster   — cluster size per Wolff update
//   /prod/obs/s           — S_full
//   /prod/obs/m2          — |M|^2 / V^2 (rotation-invariant)

#include <reticolo/reticolo.hpp>

#include <array>
#include <cstddef>

int main(int argc, char** argv) {
    using namespace reticolo;
    constexpr std::size_t k_n = 3;

    cli::Parser p{"on_sigma_wolff", "Wolff cluster for the O(3) sigma model"};
    auto const& L         = p.req<int>("L,size", "linear lattice extent");
    auto const& beta      = p.req<double>("beta", "inverse temperature");
    auto const& ndim      = p.opt<int>("ndim", 3, "spatial dimensions");
    auto const& n_cluster = p.opt<int>("n_cluster", 4, "Wolff updates per measurement");
    auto const& n_therm   = p.opt<int>("n_therm", 200, "thermalisation measurements");
    auto const& n_prod    = p.opt<int>("n_prod", 2000, "production measurements");
    auto const& seed      = p.opt<unsigned long long>("seed", 42ULL, "RNG seed");
    auto const& outpath =
        p.opt<std::string>("out", std::string{"on_sigma_wolff.h5"}, "HDF5 output path");
    if (!p.parse(argc, argv))
        return 0;

    log::start(outpath);

    Lattice<std::array<double, k_n>>::SizeVec shape(static_cast<std::size_t>(ndim),
                                                    static_cast<std::size_t>(L));
    Lattice<std::array<double, k_n>> phi{shape};
    FastRng rng{seed};
    act::OnSigma<k_n> on{.beta = beta};
    log::act(on);

    for (Site const x : phi.sites()) {
        phi[x] = {1.0, 0.0, 0.0};
    }

    io::Writer out{outpath, argc, argv, &p};
    out.start_phase("therm");
    out.start_phase("prod");
    auto cluster_therm = out.series<std::size_t>("/therm/stats/cluster");
    auto cluster_prod  = out.series<std::size_t>("/prod/stats/cluster");
    auto s_prod        = out.series<double>("/prod/obs/s");
    auto m2_prod       = out.series<double>("/prod/obs/m2");

    alg::Wolff<act::OnSigma<k_n>, FastRng> wolff{on, phi, rng};
    log::algo(wolff);

    log::info("wolf", "therm  {} measurements × {} clusters", n_therm, n_cluster);
    for (int i = 0; i < n_therm; ++i) {
        for (int k = 0; k < n_cluster; ++k) {
            cluster_therm.append(wolff.update(log::Mode::silent).cluster_size);
        }
    }
    log::info("wolf", "prod   {} measurements × {} clusters", n_prod, n_cluster);
    for (int i = 0; i < n_prod; ++i) {
        for (int k = 0; k < n_cluster; ++k) {
            cluster_prod.append(wolff.update().cluster_size);
        }
        s_prod.append(on.s_full(phi));
        m2_prod.append(obs::vector_magnetization_sq<k_n>(phi));
    }
}
