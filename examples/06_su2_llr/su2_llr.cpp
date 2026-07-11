// LLR (Gaussian-penalty) with replica exchange for SU(2) Wilson gauge theory.
//
// Mirrors u1_llr.cpp but on `MatrixLinkLattice<SU2, double>` with the generic
// `Wilson<SU2>` action. Each replica is cold-initialised to the SU(2) identity
// (all matrices = I) since the default field ctor leaves zeros, which would
// be an invalid group element.

#include <reticolo/reticolo.hpp>

#include <cstddef>
#include <filesystem>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Group  = math::group::SU2;
    using Action = action::Wilson<Group, double>;
    using Field  = MatrixLinkLattice<Group, double>;

    // ---- CLI ----
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
    auto const& workspace =
        p.opt<std::string>("workspace", std::string{"."}, "workspace folder (output + logs)");
    auto const& outfile = p.opt<std::string>(
        "out", std::string{"su2_llr.h5"}, "HDF5 output file name, inside workspace");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    log::start(workspace, outfile, /*replicas=*/true);
    std::string const outpath = (std::filesystem::path{workspace} / outfile).string();

    // ---- Base action ----
    Field::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(L));
    Action const base{.beta = beta};
    log::act(base);

    // ---- Orchestrator: owns geometry, the replica ladder, threading ----
    orch::llr::Orchestrator<Action, FastRng, updater::integ::Omelyan2, double, Field> llr{
        base,
        orch::llr::Spec{.shape      = shape,
                        .seed       = seed,
                        .e_min      = e_min,
                        .e_max      = e_max,
                        .delta      = delta,
                        .tau        = tau,
                        .n_md       = n_md,
                        .n_nr       = n_nr,
                        .n_therm_nr = n_therm_nr,
                        .n_meas_nr  = n_meas_nr,
                        .n_rm       = n_rm,
                        .n_therm_rm = n_therm_rm,
                        .n_meas_rm  = n_meas_rm}};

    // ---- Output ----
    io::Writer out{outpath, argc, argv, &p};
    out.start_phase("llr");
    llr.setup(out);

    // ---- Cold-start each replica's field to SU(2) identity (Re U_{00} =
    //      Re U_{11} = 1, all else 0); the default ctor leaves invalid zeros. ----
    for (auto& r : llr.replicas()) {
        Field& phi           = r->field();
        std::size_t const ns = phi.nsites();
        for (std::size_t mu = 0; mu < static_cast<std::size_t>(ndim); ++mu) {
            double* const blk = phi.mu_block_data(mu);
            for (std::size_t s = 0; s < ns; ++s) {
                blk[(0 * ns) + s] = 1.0;
                blk[(6 * ns) + s] = 1.0;
            }
        }
    }

    // ---- Drive: NR warm-up + RM + exchange ----
    llr.run();
}
