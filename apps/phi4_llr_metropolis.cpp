// LLR (Gaussian-penalty) with replica exchange for the phi^4 scalar field,
// sampled by LOCAL METROPOLIS instead of HMC.
//
// The only difference from phi4_llr.cpp is ONE template argument on the SAME
// orchestrator — `updater::MetropolisSampler` in the slot that otherwise holds
// the HMC sampler and its integrator — plus --sigma in place of --tau/--n_md.
// Everything else (the replica ladder, the warm-up ramp, Newton-Raphson,
// Robbins-Monro, exchange, checkpoint/resume, the output schema) is shared code.
//
// Windowed sampling is where the two updaters differ most sharply in shape. The
// Gaussian penalty (Q−E_n)²/2δ² is quadratic in a GLOBAL scalar, so a single-site
// move's ΔS depends on the current Q and the checkerboard's independence
// argument fails. The windowed sweep is therefore SEQUENTIAL with a running Q
// (see action::WindowedAction::metropolis_stencil) — which costs nothing here,
// since an LLR replica already runs its lattice passes serially inside the
// replica-parallel team.
//
// COUNTS ARE NOT COMPARABLE with the HMC app at the same number: n_therm/n_meas
// count SWEEPS here and TRAJECTORIES there, and one trajectory is n_md lattice
// passes.
//
// Output schema (HDF5): identical to phi4_llr.
//   /cfg@n_rep, /cfg@delta, /cfg@E_min, /cfg@E_max, /cfg@dE
//   /cfg/E_n                  — series, length n_rep
//   /replica_NNN/a            — series, one append per NR iter + per RM sweep
//   /replica_NNN/dE           — series, paired with /a
//   /exchange/accepted        — series, one append per RM sweep (count, 0..n_rep/2)

#include <reticolo/reticolo.hpp>

#include <cstddef>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Action = act::Phi4<double>;

    // ---- CLI ----
    cli::Parser p{"phi4_llr_metropolis",
                  "LLR (Gaussian-penalty) with replica exchange for phi^4, local Metropolis"};
    auto const cf      = app::common_flags(p, {.L = 8, .out = "phi4_llr_metropolis.h5"});
    auto const& ndim   = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& kappa  = p.opt<double>("kappa", 0.18, "hopping parameter");
    auto const& lambda = p.opt<double>("lambda", 1.0, "quartic coupling");
    auto const& e_min  = p.opt<double>("E_min", 550.0, "lower window centre");
    auto const& e_max  = p.opt<double>("E_max", 1070.0, "upper window centre");
    auto const& delta  = p.opt<double>(
        "delta", 60.0, "Gaussian penalty width δ in (S−E_n)²/2δ² (also the a-update scale)");
    auto const& spacing = p.opt<double>(
        "spacing", 0.0, "replica energy interval between window centres; 0 ⇒ equal to delta");
    auto const& sigma      = p.opt<double>("sigma", 0.5, "Metropolis proposal width");
    auto const& n_nr       = p.opt<int>("n_nr", 6, "Newton-Raphson warm-up iterations");
    auto const& n_therm_nr = p.opt<int>("n_therm_nr", 200, "thermalisation sweeps per NR iter");
    auto const& n_meas_nr  = p.opt<int>("n_meas_nr", 1000, "measurement sweeps per NR iter");
    auto const& n_rm       = p.opt<int>("n_rm", 20, "Robbins-Monro sweeps");
    auto const& n_therm_rm = p.opt<int>("n_therm_rm", 100, "thermalisation sweeps per RM sweep");
    auto const& n_meas_rm  = p.opt<int>("n_meas_rm", 500, "measurement sweeps per RM sweep");
    auto const rf          = app::llr_run_flags(p);
    if (!p.parse(argc, argv)) {
        return 0;
    }

    io::Writer out = app::open_writer(p, cf, argc, argv, /*replicas=*/true);

    // ---- Base action ----
    Lattice<double>::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(cf.L));
    Action const base{.kappa = kappa, .lambda = lambda};
    log::act(base);

    // ---- Orchestrator: owns geometry, the replica ladder, threading, resume ----
    using Llr = orch::llr::Orchestrator<Action, FastRng, updater::MetropolisSampler>;
    Llr llr{base,
            Llr::Spec{.shape            = shape,
                      .seed             = cf.seed,
                      .e_min            = e_min,
                      .e_max            = e_max,
                      .delta            = delta,
                      .spacing          = spacing,
                      .sampler          = {.sigma = sigma, .slabs_per_thread = rf.slabs},
                      .n_nr             = n_nr,
                      .n_therm_nr       = n_therm_nr,
                      .n_meas_nr        = n_meas_nr,
                      .n_rm             = n_rm,
                      .n_therm_rm       = n_therm_rm,
                      .n_meas_rm        = n_meas_rm,
                      .warm_therm       = rf.warm_therm,
                      .warm_max_traj    = rf.warm_max_traj,
                      .replica_threads  = rf.replica_threads,
                      .checkpoint_path  = rf.checkpoint,
                      .resume           = rf.resume,
                      .checkpoint_every = rf.checkpoint_every}};

    // ---- Output ----
    out.start_phase("llr");

    // ---- Drive: setup (build + resume) then NR warm-up + RM + exchange ----
    llr.setup(out);
    llr.run();
}
