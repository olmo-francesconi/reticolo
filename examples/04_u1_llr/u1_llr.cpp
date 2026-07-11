// LLR (Gaussian-penalty) with replica exchange for compact U(1) gauge theory.
//
// Energy variable: E(theta) = S_base(theta) (full Wilson action).
// Sampler:        Gauge HMC with templated integrator (default Omelyan2).
// Update:         Newton-Raphson warm-up then restarted Robbins-Monro.
// Geometry:       n_rep replicas at E_n = E_min + n * delta.
// Exchange:       even/odd alternating nearest-neighbour swaps.

#include <reticolo/reticolo.hpp>

#include <cstddef>
#include <filesystem>
#include <numbers>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Action = action::Wilson<math::group::U1, double>;
    using Field  = MatrixLinkLattice<math::group::U1, double>;

    // ---- CLI ----
    cli::Parser p{"u1_llr", "LLR with replica exchange for compact U(1) Wilson action"};
    auto const& L     = p.opt<int>("L,size", 4, "linear lattice extent");
    auto const& ndim  = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& beta  = p.opt<double>("beta", 1.0, "Wilson coupling");
    auto const& e_min = p.opt<double>("E_min", 200.0, "lower window centre");
    auto const& e_max = p.opt<double>("E_max", 1400.0, "upper window centre");
    auto const& delta = p.opt<double>(
        "delta",
        200.0,
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
    auto const& exchange  = p.opt<int>(
        "exchange", 1, "enable even/odd nearest-neighbour replica exchange in the RM phase (0/1)");
    auto const& seed = p.opt<unsigned long long>("seed", 42ULL, "RNG seed");
    auto const& workspace =
        p.opt<std::string>("workspace", std::string{"."}, "workspace folder (output + logs)");
    auto const& outfile = p.opt<std::string>(
        "out", std::string{"u1_llr.h5"}, "HDF5 output file name, inside workspace");
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
                        .n_meas_rm  = n_meas_rm,
                        .exchange   = (exchange != 0)}};

    // ---- Output ----
    io::Writer out{outpath, argc, argv, &p};
    out.start_phase("llr");
    llr.setup(out);

    // ---- Hot start: randomise each replica's link angles; the warm-up phase
    //      then drives each into its E_n window. ----
    {
        constexpr double k_hot_sigma = std::numbers::pi;  // ~uniform theta
        auto& reps                   = llr.replicas();
        std::size_t const n_rep_u    = reps.size();
#pragma omp parallel for schedule(dynamic, 1)
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            auto _ = log::scope(reps[n]->id());
            reps[n]->hot_start(k_hot_sigma);
        }
    }

    // ---- Drive: NR warm-up + RM + (optional) exchange ----
    llr.run();
}
