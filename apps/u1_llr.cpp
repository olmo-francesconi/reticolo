// LLR (Gaussian-penalty) with replica exchange for compact U(1) gauge theory.
//
// Energy variable: E(theta) = S_base(theta) (full Wilson action).
// Sampler:        Gauge HMC with templated integrator (default Omelyan2).
// Update:         Newton-Raphson warm-up then restarted Robbins-Monro.
// Geometry:       n_rep replicas at E_n = E_min + n * delta.
// Exchange:       even/odd alternating nearest-neighbour swaps.

#include <reticolo/reticolo.hpp>

#include <cstddef>
#include <numbers>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Action = action::Wilson<math::group::U1, double>;
    using Field  = MatrixLinkLattice<math::group::U1, double>;

    // ---- CLI ----
    cli::Parser p{"u1_llr", "LLR with replica exchange for compact U(1) Wilson action"};
    auto const cf     = app::common_flags(p, {.L = 4, .out = "u1_llr.h5"});
    auto const& ndim  = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& beta  = p.opt<double>("beta", 1.0, "Wilson coupling");
    auto const& e_min = p.opt<double>("E_min", 360.0, "lower window centre");
    auto const& e_max = p.opt<double>("E_max", 700.0, "upper window centre");
    auto const& delta = p.opt<double>(
        "delta", 40.0, "Gaussian penalty width δ in (S−E_n)²/2δ² (also the a-update scale)");
    auto const& spacing = p.opt<double>(
        "spacing", 0.0, "replica energy interval between window centres; 0 ⇒ equal to delta");
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
    auto const& exchange  = p.opt<int>(
        "exchange", 1, "enable even/odd nearest-neighbour replica exchange in the RM phase (0/1)");
    auto const rf = app::llr_run_flags(p);
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
                        .exchange         = (exchange != 0),
                        .replica_threads  = rf.replica_threads,
                        .slabs            = rf.slabs,
                        .checkpoint_path  = rf.checkpoint,
                        .resume           = rf.resume,
                        .checkpoint_every = rf.checkpoint_every}};

    io::Writer out{outpath, argc, argv, &p};
    out.start_phase("llr");

    // ---- Build replicas + resume ----
    llr.setup(out);

    // ---- Hot start (fresh runs only): randomise each replica's link angles;
    //      the warm-up phase then drives each into its E_n window. A resume
    //      restores already-warmed fields from the checkpoint. ----
    if (!llr.resuming()) {
        constexpr double k_hot_sigma = std::numbers::pi;  // ~uniform theta
        auto& reps                   = llr.replicas();
        std::size_t const n_rep_u    = reps.size();
#pragma omp parallel for schedule(dynamic, 1)
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            reps[n]->hot_start(k_hot_sigma);
        }
    }

    // ---- Drive: NR warm-up + RM + (optional) exchange ----
    llr.run();
}
