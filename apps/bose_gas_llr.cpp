// LLR (Gaussian-penalty) for the 4D self-interacting relativistic lattice
// Bose gas at finite chemical potential. Phase-quenched HMC samples
// `S_R = base.s_full(phi)`; the LLR window constrains the imaginary action
// observable `S_I = base.s_imag(phi)` via the `HasImagPart` dispatch in
// llr::WindowedAction.
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

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <format>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Action   = act::BoseGas<double>;
    using ReplicaT = llr::Replica<Action, FastRng>;

    // ---- CLI ----
    cli::Parser p{"bose_gas_llr", "LLR for the 4D Bose gas at finite chemical potential"};
    auto const cf      = app::common_flags(p, {.L = 4, .out = "bose_gas_llr.h5"});
    auto const& ndim   = p.opt<int>("ndim", 4, "spacetime dimensions (4 in paper)");
    auto const& mass   = p.opt<double>("mass", 1.0, "bare mass m");
    auto const& lambda = p.opt<double>("lambda", 1.0, "quartic coupling lambda");
    auto const& mu     = p.opt<double>("mu", 1.0, "chemical potential mu");
    auto const& e_min  = p.opt<double>("E_min", -10.0, "lower S_I window centre");
    auto const& e_max  = p.opt<double>("E_max", 10.0, "upper S_I window centre");
    auto const& delta  = p.opt<double>(
        "delta", 2.0, "Gaussian penalty width δ in (S_I−E_n)²/2δ² (also the a-update scale)");
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

    // ---- Replica geometry ----
    double const d_e = spacing > 0.0 ? spacing : delta;
    int const n_rep  = std::max(2, static_cast<int>(std::lround((e_max - e_min) / d_e)) + 1);
    double const e_max_snapped = e_min + (static_cast<double>(n_rep - 1) * d_e);

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
            alg::HmcSpec{.tau = tau, .n_md = n_md}));
    }

    // ---- Output ----
    FastRng exch_rng{cf.seed};
    io::Writer out{outpath, argc, argv, &p};
    out.start_phase("llr");
    out.attr<double>("/cfg@mu", mu);

    // ---- Warm-up: windowed HMC into S_I window ----
    // Hot-start every replica with an independent random field, then run this
    // replica's own windowed HMC until it is inside its E_n window. The
    // windowed force (base phase-quenched force + the S_I constraint window)
    // pins trajectories toward E_n; deep in the S_I tail keep the integrator
    // stable with enough MD steps (HmcSpec n_md).
    constexpr double k_hot_sigma   = 0.5;
    constexpr int k_warm_batches   = 50;
    constexpr int k_warm_batch_len = 10;
    std::size_t const n_rep_u      = static_cast<std::size_t>(n_rep);
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t n = 0; n < n_rep_u; ++n) {
        auto _ = log::scope(reps[n]->id());
        reps[n]->hot_start(k_hot_sigma);
        reps[n]->warm_into_window(k_warm_batches, k_warm_batch_len, 1.0);
    }

    // ---- Drive: NR warm-up + RM + exchange ----
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
