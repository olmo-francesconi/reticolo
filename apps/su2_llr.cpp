// LLR (Gaussian-penalty) with replica exchange for SU(2) Wilson gauge theory.
//
// Mirrors u1_llr.cpp but on `MatrixLinkLattice<SU2, double>` with the generic
// `Wilson<SU2>` action. Each replica is cold-initialised to the SU(2) identity
// (all matrices = I) since the default field ctor leaves zeros, which would
// be an invalid group element.

#include <reticolo/reticolo.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Group    = gauge_group::SU2;
    using Action   = action::Wilson<Group, double>;
    using Field    = MatrixLinkLattice<Group, double>;
    using ReplicaT = llr::Replica<Action, FastRng, alg::integ::Omelyan2, double, Field>;

    cli::Parser p{"su2_llr", "LLR with replica exchange for SU(2) Wilson action"};
    auto const& L     = p.opt<int>("L,size", 4, "linear lattice extent");
    auto const& ndim  = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& beta  = p.opt<double>("beta", 2.3, "Wilson coupling");
    auto const& e_min = p.opt<double>("E_min", 200.0, "lower window centre");
    auto const& e_max = p.opt<double>("E_max", 1400.0, "upper window centre");
    auto const& delta = p.opt<double>(
        "delta", 200.0, "single LLR tuning knob: Gaussian half-width AND replica spacing.");
    auto const& tau  = p.opt<double>("tau", 1.0, "HMC trajectory length");
    auto const& n_md = p.opt<int>("n_md", 20, "MD steps per trajectory");
    auto const& n_nr = p.opt<int>("n_nr", 6, "Newton-Raphson warm-up iterations");
    auto const& n_therm_nr =
        p.opt<int>("n_therm_nr", 200, "thermalisation trajectories per NR iter");
    auto const& n_meas_nr = p.opt<int>("n_meas_nr", 1000, "measurement trajectories per NR iter");
    auto const& n_rm      = p.opt<int>("n_rm", 20, "Robbins-Monro sweeps");
    auto const& n_therm_rm =
        p.opt<int>("n_therm_rm", 100, "thermalisation trajectories per RM sweep");
    auto const& n_meas_rm = p.opt<int>("n_meas_rm", 500, "measurement trajectories per RM sweep");
    auto const& seed      = p.opt<unsigned long long>("seed", 42ULL, "RNG seed");
    auto const& outpath   = p.opt<std::string>("out", std::string{"su2_llr.h5"}, "HDF5 output");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    log::start(outpath);

    Field::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(L));
    Action const base{.beta = beta};
    log::act(base);

    int const n_rep  = std::max(2, static_cast<int>(std::lround((e_max - e_min) / delta)) + 1);
    double const d_e = delta;
    double const e_max_snapped = e_min + (static_cast<double>(n_rep - 1) * d_e);

    std::vector<std::unique_ptr<ReplicaT>> reps;
    reps.reserve(static_cast<std::size_t>(n_rep));
    for (int n = 0; n < n_rep; ++n) {
        double const e_n = e_min + (static_cast<double>(n) * d_e);
        reps.push_back(std::make_unique<ReplicaT>(
            base,
            FastRng{seed + 1ULL + static_cast<unsigned long long>(n)},
            ReplicaT::Spec{
                .id = std::format("r{:03}", n), .shape = shape, .e_n = e_n, .delta = delta},
            alg::HmcSpec{.tau = tau, .n_md = n_md}));
        // Cold-start each replica's field to SU(2) identity (Re U_{00} =
        // Re U_{11} = 1, all else 0).
        Field& phi           = reps.back()->phi();
        std::size_t const ns = phi.nsites();
        for (std::size_t mu = 0; mu < static_cast<std::size_t>(ndim); ++mu) {
            double* const blk = phi.mu_block_data(mu);
            for (std::size_t s = 0; s < ns; ++s) {
                blk[(0 * ns) + s] = 1.0;
                blk[(6 * ns) + s] = 1.0;
            }
        }
    }

    FastRng exch_rng{seed};
    io::Writer out{outpath, argc, argv, &p};
    out.start_phase("llr");

    llr::run(reps,
             exch_rng,
             llr::DriverSpec{.n_nr       = n_nr,
                             .n_therm_nr = n_therm_nr,
                             .n_meas_nr  = n_meas_nr,
                             .n_rm       = n_rm,
                             .n_therm_rm = n_therm_rm,
                             .n_meas_rm  = n_meas_rm,
                             .delta      = delta,
                             .e_min      = e_min,
                             .E_max      = e_max_snapped,
                             .d_e        = d_e},
             out);
}
