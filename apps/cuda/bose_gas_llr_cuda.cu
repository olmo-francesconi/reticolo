// Complex-LLR (Gaussian-penalty) with replica exchange for the 4D relativistic
// lattice Bose gas at finite chemical potential, on the CUDA backend. The GPU
// twin of bose_gas_llr.cpp: phase-quenched HMC samples S_R = base.s_full, and the
// LLR window constrains the imaginary observable S_I = base.s_imag (mode B of the
// device WindowedAction, selected because BoseGas exposes s_imag/compute_force_imag
// on the device). Same CLI + LLR schedule (NR warm-up → RM + even/odd exchange)
// and same param-swap output schema as bose_gas_llr.cpp / phi4_llr_cuda.
//
// Output schema (HDF5): /cfg@*, /cfg@mu, /cfg/E_n (window centres in S_I),
// /replica_NNN/{a,dE,E_n} (dE = <S_I − E_n>), /exchange/accepted. Feed the per-µ
// output through examples/05_bose_gas_llr/analyze.py. Group by E_n downstream.

#include <reticolo/cuda/cuda.hpp>
#include <reticolo/cuda/llr/driver.hpp>
#include <reticolo/cuda/llr/replica.hpp>
#include <reticolo/reticolo.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    using namespace reticolo;
    using DField   = cuda::DeviceField<cplx<double>>;
    using ReplicaT = cuda::llr::Replica<act::BoseGas<double>, alg::integ::Omelyan2, DField>;

    // ---- CLI (mirrors bose_gas_llr.cpp) ----
    cli::Parser p{"bose_gas_llr_cuda", "Complex-LLR + replica exchange for the 4D Bose gas (CUDA)"};
    auto const cf      = app::common_flags(p, {.L = 4, .out = "bose_gas_llr_cuda.h5"});
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
    std::vector<std::size_t> const shape(static_cast<std::size_t>(ndim),
                                         static_cast<std::size_t>(cf.L));
    act::BoseGas<double> const base{.mass = mass, .lambda = lambda, .mu = mu};
    log::act(base);

    // ---- Replica geometry ----
    double const d_e = spacing > 0.0 ? spacing : delta;
    int const n_rep  = std::max(2, static_cast<int>(std::lround((e_max - e_min) / d_e)) + 1);
    double const e_max_snapped = e_min + (static_cast<double>(n_rep - 1) * d_e);

    // ---- Replicas (cold φ = 0; hot-started + warmed into their window below) ----
    std::vector<std::unique_ptr<ReplicaT>> reps;
    reps.reserve(static_cast<std::size_t>(n_rep));
    for (int n = 0; n < n_rep; ++n) {
        double const e_n = e_min + (static_cast<double>(n) * d_e);
        reps.push_back(std::make_unique<ReplicaT>(base,
                                                  shape,
                                                  e_n,
                                                  delta,
                                                  tau,
                                                  n_md,
                                                  static_cast<std::uint64_t>(cf.seed) + 1ULL +
                                                      static_cast<std::uint64_t>(n)));
    }

    // Hot-start each replica (φ ~ N(0, σ²)) before warming into its S_I window —
    // mirrors the CPU bose_gas_llr hot_start(0.5).
    constexpr double k_hot_sigma = 0.5;
    for (auto& r : reps) {
        r->hot_start(k_hot_sigma);
    }
    log::info("llr", "warm {} replicas into window", n_rep);
    cuda::llr::warm_all(reps, 50, 10, 1.0);

    // ---- Output ----
    FastRng exch_rng{cf.seed};
    io::Writer out{outpath, argc, argv, &p};
    out.start_phase("llr");
    out.attr<double>("/cfg@mu", mu);

    // ---- Drive: NR warm-up + RM + exchange ----
    cuda::llr::run(reps,
                   exch_rng,
                   cuda::llr::DriverSpec{.n_nr       = n_nr,
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
