// LLR (Gaussian-penalty) with replica exchange for SU(3) Wilson gauge theory.
//
// Mirrors su2_llr on `MatrixLinkLattice<SU3, double>` with `Wilson<SU3>`. Each
// replica is cold-initialised to the SU(3) identity (all matrices = I) since
// the default field ctor leaves zeros, an invalid group element. Default window
// brackets the action distribution of a short su3_hmc run at this geometry/beta.

#include <reticolo/reticolo.hpp>

#include <cstddef>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Group  = math::group::SU3;
    using Action = action::Wilson<Group, double>;
    using Field  = MatrixLinkLattice<Group, double>;

    // ---- CLI ----
    cli::Parser p{"su3_llr", "LLR with replica exchange for SU(3) Wilson action"};
    auto const cf     = app::common_flags(p, {.L = 4, .out = "su3_llr.h5"});
    auto const& ndim  = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& beta  = p.opt<double>("beta", 6.0, "Wilson coupling");
    auto const& e_min = p.opt<double>("E_min", 3420.0, "lower window centre");
    auto const& e_max = p.opt<double>("E_max", 4000.0, "upper window centre");
    auto const& delta = p.opt<double>(
        "delta", 70.0, "Gaussian penalty width δ in (S−E_n)²/2δ² (also the a-update scale)");
    auto const& spacing = p.opt<double>(
        "spacing", 0.0, "replica energy interval between window centres; 0 ⇒ equal to delta");
    auto const& tau  = p.opt<double>("tau", 1.0, "HMC trajectory length");
    auto const& n_md = p.opt<int>("n_md", 30, "MD steps per trajectory");
    auto const& n_nr = p.opt<int>("n_nr", 6, "Newton-Raphson warm-up iterations");
    auto const& n_therm_nr =
        p.opt<int>("n_therm_nr", 200, "thermalisation trajectories per NR iter");
    auto const& n_meas_nr = p.opt<int>("n_meas_nr", 1000, "measurement trajectories per NR iter");
    auto const& n_rm      = p.opt<int>("n_rm", 20, "Robbins-Monro sweeps");
    auto const& n_therm_rm =
        p.opt<int>("n_therm_rm", 100, "thermalisation trajectories per RM sweep");
    auto const& n_meas_rm = p.opt<int>("n_meas_rm", 500, "measurement trajectories per RM sweep");
    auto const rf         = app::llr_run_flags(p);
    if (!p.parse(argc, argv)) {
        return 0;
    }

    log::start(cf.workspace, cf.out, /*replicas=*/true);
    std::string const outpath = app::out_path(cf);

    // ---- Base action ----
    Field::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(cf.L));
    Action const base{.beta = beta};
    log::act(base);

    // ---- Orchestrator: owns geometry, the replica ladder, threading, resume ----
    orch::llr::Orchestrator<Action, FastRng, updater::integ::Omelyan2, double, Field> llr{
        base,
        orch::llr::Spec{.shape            = shape,
                        .seed             = cf.seed,
                        .e_min            = e_min,
                        .e_max            = e_max,
                        .delta            = delta,
                        .spacing          = spacing,
                        .tau              = tau,
                        .n_md             = n_md,
                        .n_nr             = n_nr,
                        .n_therm_nr       = n_therm_nr,
                        .n_meas_nr        = n_meas_nr,
                        .n_rm             = n_rm,
                        .n_therm_rm       = n_therm_rm,
                        .n_meas_rm        = n_meas_rm,
                        .replica_threads  = rf.replica_threads,
                        .slabs            = rf.slabs,
                        .checkpoint_path  = rf.checkpoint,
                        .resume           = rf.resume,
                        .checkpoint_every = rf.checkpoint_every}};

    // ---- Output ----
    io::Writer out{outpath, argc, argv, &p};
    out.start_phase("llr");

    // ---- Build replicas + resume ----
    llr.setup(out);

    // ---- Cold-start fresh fields to SU(3) identity (Re U_{ii} = 1); the default
    //      field ctor leaves zeros, an invalid group element. A resume restores
    //      valid fields instead. ----
    if (!llr.resuming()) {
        for (auto& r : llr.replicas()) {
            Field& phi           = r->field();
            std::size_t const ns = phi.nsites();
            for (std::size_t mu = 0; mu < static_cast<std::size_t>(ndim); ++mu) {
                double* const blk = phi.mu_block_data(mu);
                for (std::size_t s = 0; s < ns; ++s) {
                    blk[(0 * ns) + s]  = 1.0;  // Re U_{00}
                    blk[(8 * ns) + s]  = 1.0;  // Re U_{11}
                    blk[(16 * ns) + s] = 1.0;  // Re U_{22}
                }
            }
        }
    }

    // ---- Drive: NR warm-up + RM + exchange ----
    llr.run();
}
