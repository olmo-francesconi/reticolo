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
// the per-µ output of this binary through examples/05_bose_gas_llr/analyze.py.

#include <reticolo/reticolo.hpp>

#include <complex>
#include <cstddef>
#include <filesystem>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Action = act::BoseGas<double>;

    // ---- CLI ----
    cli::Parser p{"bose_gas_llr", "LLR for the 4D Bose gas at finite chemical potential"};
    auto const& L      = p.opt<int>("L,size", 4, "linear lattice extent");
    auto const& ndim   = p.opt<int>("ndim", 4, "spacetime dimensions (4 in paper)");
    auto const& mass   = p.opt<double>("mass", 1.0, "bare mass m");
    auto const& lambda = p.opt<double>("lambda", 1.0, "quartic coupling lambda");
    auto const& mu     = p.opt<double>("mu", 1.0, "chemical potential mu");
    auto const& e_min  = p.opt<double>("E_min", -10.0, "lower S_I window centre");
    auto const& e_max  = p.opt<double>("E_max", 10.0, "upper S_I window centre");
    auto const& delta =
        p.opt<double>("delta",
                      2.0,
                      "Gaussian window half-width in S_I (and the replica spacing unless "
                      "--spacing is given).");
    auto const& spacing =
        p.opt<double>("spacing",
                      0.0,
                      "window-centre spacing in S_I; 0 means `delta` (non-overlapping). "
                      "spacing < delta gives overlapping windows: denser constraint grid "
                      "without stiffening the window force. n_rep is derived from it.");
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
    auto const& seed      = p.opt<unsigned long long>("seed", 42ULL, "RNG seed");
    auto const& workspace =
        p.opt<std::string>("workspace", std::string{"."}, "workspace folder (output + logs)");
    auto const& outfile = p.opt<std::string>(
        "out", std::string{"bose_gas_llr.h5"}, "HDF5 output file name, inside workspace");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    log::start(workspace, outfile, /*replicas=*/true);
    std::string const outpath = (std::filesystem::path{workspace} / outfile).string();

    // ---- Base action ----
    Lattice<std::complex<double>>::SizeVec shape(static_cast<std::size_t>(ndim),
                                                 static_cast<std::size_t>(L));
    Action const base{.mass = mass, .lambda = lambda, .mu = mu};
    log::act(base);

    // ---- Orchestrator: owns geometry, the replica ladder, threading ----
    orch::llr::Orchestrator<Action, FastRng> llr{base,
                                                 orch::llr::Spec{.shape      = shape,
                                                                 .seed       = seed,
                                                                 .e_min      = e_min,
                                                                 .e_max      = e_max,
                                                                 .delta      = delta,
                                                                 .spacing    = spacing,
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
    out.attr<double>("/cfg@mu", mu);
    llr.setup(out);

    // ---- Hot start: give every replica an independent random field; the warm-up
    //      phase then drives each into its E_n (S_I) window. ----
    {
        constexpr double k_hot_sigma = 0.5;
        auto& reps                   = llr.replicas();
        std::size_t const n_rep_u    = reps.size();
#pragma omp parallel for schedule(dynamic, 1)
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            auto _ = log::scope(reps[n]->id());
            reps[n]->hot_start(k_hot_sigma);
        }
    }

    // ---- Drive: NR warm-up + RM + exchange ----
    llr.run();
}
