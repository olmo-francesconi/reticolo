// LLR (Gaussian-penalty) for the 4D self-interacting relativistic lattice
// Bose gas at finite chemical potential. Phase-quenched HMC samples
// `S_R = base.s_full(phi)`; the LLR window constrains the imaginary action
// observable `S_I = base.s_imag(phi)` via the `HasImagPart` dispatch in
// action::WindowedAction.
//
// Output schema (HDF5):
//  /cfg@n_rep, /cfg@delta, /cfg@E_min, /cfg@E_max, /cfg@dE, /cfg@mu
//  /cfg/E_n              — n_rep values (window centres in S_I)
//  /replica_NNN/a        — series, one append per NR iter + per RM sweep
//  /replica_NNN/dE       — series, paired with /a (<S_I − E_n>)
//  /exchange/accepted    — series, one int per RM sweep
//
// arxiv:1910.11026 reproduces the paper's `<e^{iφ}>_pq(μ)` curve by feeding
// the per-µ output of this binary through examples/06_bose_gas_llr/analyze.py.

#include <reticolo/reticolo.hpp>

#include <complex>
#include <cstddef>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Action = act::BoseGas<double>;

    // ---- CLI ----
    cli::Parser p{"bose_gas_llr", "LLR for the 4D Bose gas at finite chemical potential"};
    auto const cf      = app::common_flags(p, {.L = 4, .out = "bose_gas_llr.h5"});
    auto const& ndim   = p.opt<int>("ndim", 4, "spacetime dimensions (4 in paper)");
    auto const& mass   = p.opt<double>("mass", 1.0, "bare mass m");
    auto const& lambda = p.opt<double>("lambda", 1.0, "quartic coupling lambda");
    auto const& mu     = p.opt<double>("mu", 1.0, "chemical potential mu");
    auto const& e_min  = p.opt<double>("E_min", -11.0, "lower S_I window centre");
    auto const& e_max  = p.opt<double>("E_max", 11.0, "upper S_I window centre");
    auto const& delta  = p.opt<double>(
        "delta", 2.5, "Gaussian penalty width δ in (S_I−E_n)²/2δ² (also the a-update scale)");
    auto const& spacing =
        p.opt<double>("spacing",
                      0.0,
                      "replica energy interval between window centres in S_I; 0 ⇒ equal to delta");
    auto const& tau  = p.opt<double>("tau", 1.0, "HMC trajectory length");
    auto const& n_md = p.opt<int>("n_md", 10, "MD steps per trajectory");
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
    Lattice<std::complex<double>>::SizeVec shape(static_cast<std::size_t>(ndim),
                                                 static_cast<std::size_t>(cf.L));
    Action const base{.mass = mass, .lambda = lambda, .mu = mu};
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
    out.attr<double>("/cfg@mu", mu);

    // ---- Build replicas + resume ----
    llr.setup(out);

    // ---- Hot start (fresh runs only): give every replica an independent random
    //      field; the warm-up phase then drives each into its E_n (S_I) window.
    //      A resume restores already-warmed fields. ----
    if (!llr.resuming()) {
        constexpr double k_hot_sigma = 0.5;
        auto& reps                   = llr.replicas();
        std::size_t const n_rep_u    = reps.size();
#pragma omp parallel for schedule(dynamic, 1)
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            reps[n]->hot_start(k_hot_sigma);
        }
    }

    // ---- Drive: NR warm-up + RM + exchange ----
    llr.run();
}
