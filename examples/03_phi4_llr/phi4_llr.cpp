// LLR (Gaussian-penalty) with replica exchange for the phi^4 scalar field.
//
// Energy variable: E(phi) = S_base(phi) (the full action).
// Sampler:        HMC with a templated integrator (default Omelyan2).
// Update:         Newton-Raphson warm-up (n_nr iters), then restarted RM
//                 with counter k reset to 0 after the warm-up.
// Geometry:       n_rep replicas at E_n = E_min + n * dE,
//                 dE = (E_max - E_min) / (n_rep - 1).
// Exchange:       even/odd alternating nearest-neighbour swaps after each RM sweep.
//
// Output schema (HDF5):
//   /cfg@n_rep, /cfg@delta, /cfg@E_min, /cfg@E_max, /cfg@dE
//   /cfg/E_n                  — series, length n_rep
//   /replica_NNN/a            — series, one append per NR iter + per RM sweep
//   /replica_NNN/dE           — series, paired with /a
//   /exchange/accepted        — series, one append per RM sweep (count, 0..n_rep/2)

#include <reticolo/reticolo.hpp>

#include <cstddef>
#include <filesystem>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Action = act::Phi4<double>;

    // ---- CLI ----
    cli::Parser p{"phi4_llr", "LLR (Gaussian-penalty) with replica exchange for phi^4"};
    auto const& L      = p.opt<int>("L,size", 8, "linear lattice extent");
    auto const& ndim   = p.opt<int>("ndim", 3, "spatial dimensions");
    auto const& kappa  = p.opt<double>("kappa", 0.18, "hopping parameter");
    auto const& lambda = p.opt<double>("lambda", 1.0, "quartic coupling");
    auto const& e_min  = p.opt<double>("E_min", -100.0, "lower window centre");
    auto const& e_max  = p.opt<double>("E_max", 100.0, "upper window centre");
    auto const& delta  = p.opt<double>(
        "delta",
        25.0,
        "single LLR tuning knob: Gaussian half-width AND replica spacing. "
        "n_rep is derived so that adjacent window centres are exactly `delta` apart.");
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
        "out", std::string{"phi4_llr.h5"}, "HDF5 output file name, inside workspace");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    log::start(workspace, outfile, /*replicas=*/true);
    std::string const outpath = (std::filesystem::path{workspace} / outfile).string();

    // ---- Base action ----
    Lattice<double>::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(L));
    Action const base{.kappa = kappa, .lambda = lambda};
    log::act(base);

    // ---- Orchestrator: owns geometry, the replica ladder, threading ----
    orch::llr::Orchestrator<Action, FastRng> llr{base,
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

    // ---- Drive: setup then NR warm-up + RM + exchange ----
    llr.setup(out);
    llr.run();
}
