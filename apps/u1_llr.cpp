// LLR (Gaussian-penalty) with replica exchange for compact U(1) gauge theory.
//
// Energy variable: E(theta) = S_base(theta) (full Wilson action).
// Sampler:        Gauge HMC with templated integrator (default Omelyan2).
// Update:         Newton-Raphson warm-up then restarted Robbins-Monro.
// Geometry:       n_rep replicas at E_n = E_min + n * delta.
// Exchange:       even/odd alternating nearest-neighbour swaps.

#include <reticolo/reticolo.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Action   = action::Wilson<math::group::U1, double>;
    using ReplicaT = llr::Replica<Action,
                                  FastRng,
                                  alg::integ::Omelyan2,
                                  double,
                                  MatrixLinkLattice<math::group::U1, double>>;

    // ---- CLI ----
    cli::Parser p{"u1_llr", "LLR with replica exchange for compact U(1) Wilson action"};
    auto const cf     = app::common_flags(p, {.L = 4, .out = "u1_llr.h5"});
    auto const& ndim  = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& beta  = p.opt<double>("beta", 1.0, "Wilson coupling");
    auto const& e_min = p.opt<double>("E_min", 200.0, "lower window centre");
    auto const& e_max = p.opt<double>("E_max", 1400.0, "upper window centre");
    auto const& delta = p.opt<double>(
        "delta", 200.0, "Gaussian penalty width δ in (S−E_n)²/2δ² (also the a-update scale)");
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
    MatrixLinkLattice<math::group::U1, double>::SizeVec shape(static_cast<std::size_t>(ndim),
                                                              static_cast<std::size_t>(cf.L));
    Action const base{.beta = beta};
    log::act(base);

    // ---- Replica geometry ----
    double const d_e = spacing > 0.0 ? spacing : delta;
    int const n_rep  = std::max(2, static_cast<int>(std::lround((e_max - e_min) / d_e)) + 1);
    double const e_max_snapped = e_min + (static_cast<double>(n_rep - 1) * d_e);
    auto const plan            = llr::plan_threads(n_rep, rf.threads, rf.replica_threads);

    // ---- Replicas ----
    std::vector<std::unique_ptr<ReplicaT>> reps;
    reps.reserve(static_cast<std::size_t>(n_rep));
    for (int n = 0; n < n_rep; ++n) {
        double const e_n = e_min + (static_cast<double>(n) * d_e);
        reps.push_back(std::make_unique<ReplicaT>(
            base,
            FastRng{cf.seed + 1ULL + static_cast<unsigned long long>(n)},
            ReplicaT::Spec{
                .id = std::format("r{:03}", n), .shape = shape, .e_n = e_n, .delta = delta},
            alg::HmcSpec{
                .tau = tau, .n_md = n_md, .n_threads = plan.m, .slabs_per_thread = rf.slabs}));
    }

    // ---- Resume or fresh warm-up ----
    FastRng exch_rng{cf.seed};
    bool const resuming = !rf.resume.empty();
    llr::OrchState resume_state{};
    if (resuming) {
        resume_state = llr::load_ensemble(rf.resume, reps, exch_rng);
        log::info("llr",
                  "resumed from {}  phase={} iter={}",
                  rf.resume,
                  resume_state.phase,
                  resume_state.iter);
    }

    io::Writer out{outpath, argc, argv, &p};
    out.start_phase("llr");

    // ---- Warm-up: windowed HMC into S window (fresh runs only; a resume
    //      restores already-warmed fields from the checkpoint). ----
    if (!resuming) {
        // Hot-start each replica with random link angles, then run its own
        // windowed HMC until it sits inside its E_n window. The windowed force
        // pins trajectories toward E_n; deep in the S tail keep the leapfrog
        // stable with enough MD steps (HmcSpec n_md).
        constexpr double k_hot_sigma   = std::numbers::pi;  // ~uniform theta
        constexpr int k_warm_batches   = 50;
        constexpr int k_warm_batch_len = 10;
        std::size_t const n_rep_u      = static_cast<std::size_t>(n_rep);
#pragma omp parallel for schedule(dynamic, 1)
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            auto _ = log::scope(reps[n]->id());
            reps[n]->hot_start(k_hot_sigma);
            reps[n]->warm_into_window(k_warm_batches, k_warm_batch_len, 1.0);
        }
    }

    // ---- Drive: NR warm-up + RM + (optional) exchange ----
    llr::run(reps,
             exch_rng,
             llr::DriverSpec{.n_nr             = n_nr,
                             .n_therm_nr       = n_therm_nr,
                             .n_meas_nr        = n_meas_nr,
                             .n_rm             = n_rm,
                             .n_therm_rm       = n_therm_rm,
                             .n_meas_rm        = n_meas_rm,
                             .delta            = delta,
                             .e_min            = e_min,
                             .E_max            = e_max_snapped,
                             .d_e              = d_e,
                             .exchange         = (exchange != 0),
                             .replica_threads  = plan.m,
                             .slabs            = rf.slabs,
                             .concurrency      = plan.concurrency,
                             .nested           = plan.m > 1,
                             .checkpoint_path  = rf.checkpoint,
                             .checkpoint_every = rf.checkpoint_every,
                             .start_phase      = resuming ? resume_state.phase : 0,
                             .start_iter       = resuming ? resume_state.iter : 0},
             out);
}
