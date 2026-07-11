// LLR (Gaussian-penalty) with replica exchange for the sine-Gordon scalar field.
//
// Mirrors phi4_llr on a plain Lattice<double>. Energy variable E(phi) =
// S_base(phi). NOTE the default kappa=0.1 (not the sine_gordon_hmc default of
// 1.0): at d≤3 the hopping term destabilises the zero mode near kappa≈1, so a
// smaller kappa keeps the action bounded for a clean density-of-states window.
// The default window brackets the action distribution of a short
// sine_gordon_hmc run at this geometry/coupling.
//
// Output schema (HDF5):
//   /cfg@n_rep, /cfg@delta, /cfg@E_min, /cfg@E_max, /cfg@dE
//   /cfg/E_n                  — series, length n_rep
//   /replica_NNN/a            — series, one append per NR iter + per RM sweep
//   /replica_NNN/dE           — series, paired with /a
//   /exchange/accepted        — series, one append per RM sweep

#include <reticolo/reticolo.hpp>

#include <cstddef>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Action = act::SineGordon<double>;

    // ---- CLI ----
    cli::Parser p{"sine_gordon_llr",
                  "LLR (Gaussian-penalty) with replica exchange for sine-Gordon"};
    auto const cf     = app::common_flags(p, {.L = 8, .out = "sine_gordon_llr.h5"});
    auto const& ndim  = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& kappa = p.opt<double>("kappa", 0.1, "hopping parameter (kept small for stability)");
    auto const& alpha = p.opt<double>("alpha", 1.0, "cosine-potential strength");
    auto const& e_min = p.opt<double>("E_min", -2170.0, "lower window centre");
    auto const& e_max = p.opt<double>("E_max", -1800.0, "upper window centre");
    auto const& delta = p.opt<double>(
        "delta", 50.0, "Gaussian penalty width δ in (S−E_n)²/2δ² (also the a-update scale)");
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
    auto const rf         = app::llr_run_flags(p);
    if (!p.parse(argc, argv)) {
        return 0;
    }

    log::start(cf.workspace, cf.out, /*replicas=*/true);
    std::string const outpath = app::out_path(cf);

    // ---- Base action ----
    Lattice<double>::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(cf.L));
    Action const base{.kappa = kappa, .alpha = alpha};
    log::act(base);

    // ---- Orchestrator: owns geometry, the replica ladder, threading, resume ----
    orch::llr::Orchestrator<Action, FastRng> llr{
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

    // ---- Drive: setup (build + resume) then NR warm-up + RM + exchange ----
    llr.setup(out);
    llr.run();
}
