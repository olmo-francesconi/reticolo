// Generalized-observable LLR: reconstruct the density of states of an ARBITRARY
// observable Q, not the action itself. Here we sample the φ⁴ scalar field but
// window / adapt / exchange on the field amplitude Q = Σ_x φ(x)² (see
// observable.hpp). It is the ordinary replica-exchange LLR driver `orch::llr::run`
// — the ONLY change from apps/phi4_llr.cpp is the `Constraint` template argument
// on `orch::llr::Replica`; the base action, the observable, the window and the
// sampler are independent building blocks.
//
// The window is  S_win = S_φ⁴ + a·Q + (Q − E_n)²/2δ² , so the converged per-replica
// tilt a_n = d ln ρ/dQ|_{E_n} and ln ρ(Q) = ∫ a dQ (reconstructed in analyze.py).
//
// CLI window range is in PER-SITE amplitude ⟨φ²⟩ = Q/V (intuitive); the app scales
// by the volume V = L^ndim to the Σφ² units the constraint works in.
//
// Output: the standard LLR HDF5 schema (/cfg/E_n, /replica_NNN/a, /exchange/…),
// plus /cfg@V and /cfg@observable so analyze.py can label the axis.

#include <reticolo/reticolo.hpp>

#include "observable.hpp"

#include <cmath>
#include <cstddef>
#include <string>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Action     = act::Phi4<double>;
    using Constraint = action::ObservableConstraint<example::FieldAmplitude<double>>;
    using Llr        = orch::llr::Orchestrator<Action,
                                               FastRng,
                                               updater::integ::Omelyan2,
                                               double,
                                               Lattice<double>,
                                               Constraint>;

    // ---- CLI ----
    cli::Parser p{"generalized_dos_llr", "LLR density of states of the field amplitude Σφ²"};
    auto const cf      = app::common_flags(p, {.L = 8, .out = "amp_llr.h5"});
    auto const& ndim   = p.opt<int>("ndim", 3, "spatial dimensions");
    auto const& kappa  = p.opt<double>("kappa", 0.18, "hopping parameter");
    auto const& lambda = p.opt<double>("lambda", 1.0, "quartic coupling");
    // Window range + spacing in PER-SITE amplitude units ⟨φ²⟩ = Σφ²/V.
    auto const& amp_min = p.opt<double>("amp_min", 0.38, "lower window centre, ⟨φ²⟩ per site");
    auto const& amp_max = p.opt<double>("amp_max", 0.88, "upper window centre, ⟨φ²⟩ per site");
    auto const& damp    = p.opt<double>(
        "damp", 0.017, "window width δ AND replica spacing, in ⟨φ²⟩ units (the single LLR knob)");
    auto const& tau  = p.opt<double>("tau", 1.0, "HMC trajectory length");
    auto const& n_md = p.opt<int>("n_md", 20, "MD steps per trajectory");
    auto const& n_nr = p.opt<int>("n_nr", 6, "Newton-Raphson warm-up iterations");
    auto const& n_therm_nr =
        p.opt<int>("n_therm_nr", 80, "thermalisation trajectories per NR iter");
    auto const& n_meas_nr = p.opt<int>("n_meas_nr", 250, "measurement trajectories per NR iter");
    auto const& n_rm      = p.opt<int>("n_rm", 18, "Robbins-Monro sweeps");
    auto const& n_therm_rm =
        p.opt<int>("n_therm_rm", 50, "thermalisation trajectories per RM sweep");
    auto const& n_meas_rm = p.opt<int>("n_meas_rm", 250, "measurement trajectories per RM sweep");
    auto const rf         = app::llr_run_flags(p);
    if (!p.parse(argc, argv)) {
        return 0;
    }

    log::start(cf.workspace, cf.out, /*replicas=*/true);
    std::string const outpath = app::out_path(cf);

    // ---- Base action + volume ----
    Lattice<double>::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(cf.L));
    Action const base{.kappa = kappa, .lambda = lambda};
    log::act(base);
    double const volume = std::pow(static_cast<double>(cf.L), ndim);

    // ---- Window geometry (⟨φ²⟩ → Σφ² by × volume) ----
    double const e_min = volume * amp_min;
    double const e_max = volume * amp_max;
    double const delta = volume * damp;  // Gaussian width AND replica spacing (Σφ² units)

    // ---- Orchestrator: windows on Q = Σφ² via ObservableConstraint. The only
    //      change from a plain-action LLR app is the Constraint template arg. ----
    Llr llr{base,
            orch::llr::Spec{.shape           = shape,
                            .seed            = cf.seed,
                            .e_min           = e_min,
                            .e_max           = e_max,
                            .delta           = delta,
                            .tau             = tau,
                            .n_md            = n_md,
                            .n_nr            = n_nr,
                            .n_therm_nr      = n_therm_nr,
                            .n_meas_nr       = n_meas_nr,
                            .n_rm            = n_rm,
                            .n_therm_rm      = n_therm_rm,
                            .n_meas_rm       = n_meas_rm,
                            .replica_threads = rf.replica_threads,
                            .slabs           = rf.slabs}};

    // ---- Output ----
    io::Writer out{outpath, argc, argv, &p};
    out.attr<double>("/cfg@V", volume);  // so analyze.py maps Σφ² → ⟨φ²⟩
    out.attr<std::string>("/cfg@observable", "field_amplitude_sum_phi2");
    out.start_phase("llr");

    // ---- Drive: setup then warm-up + NR + RM + even/odd exchange ----
    llr.setup(out);
    llr.run();
}
