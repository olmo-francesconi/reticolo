// LLR (Gaussian-penalty) with replica exchange for phi^4 on the CUDA backend.
// The GPU twin of phi4_llr.cpp: same CLI, same LLR schedule (NR warm-up → RM +
// even/odd exchange). Model B — every replica owns its own cuda::Hmc on its own
// stream; the driver enqueues all replicas' trajectory before syncing so they
// overlap on the GPU (the analog of the CPU's omp-over-replicas).
//
// Exchange is param-swap (windows migrate across slots), so E_n is a per-slot
// series; group by E_n downstream to recover per-window DoS.
//
// Output schema (HDF5):
//   /cfg@n_rep, /cfg@delta, /cfg@E_min, /cfg@E_max, /cfg@dE
//   /cfg/E_n                  — series, initial per-slot window centres
//   /replica_NNN/a            — series, per NR iter + per RM sweep
//   /replica_NNN/dE           — series, paired with /a
//   /replica_NNN/E_n          — series, time-varying window centre (param-swap)
//   /exchange/accepted        — series, one append per RM sweep

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
    using DField   = cuda::DeviceField<double>;
    using ReplicaT = cuda::llr::Replica<act::Phi4<double>, alg::integ::Omelyan2, DField>;

    // ---- CLI (mirrors phi4_llr.cpp) ----
    cli::Parser p{"phi4_llr_cuda", "LLR (Gaussian-penalty) + replica exchange for phi^4 (CUDA)"};
    auto const cf      = app::common_flags(p, {.L = 8, .out = "phi4_llr_cuda.h5"});
    auto const& ndim   = p.opt<int>("ndim", 3, "spatial dimensions");
    auto const& kappa  = p.opt<double>("kappa", 0.18, "hopping parameter");
    auto const& lambda = p.opt<double>("lambda", 1.0, "quartic coupling");
    auto const& e_min  = p.opt<double>("E_min", -100.0, "lower window centre");
    auto const& e_max  = p.opt<double>("E_max", 100.0, "upper window centre");
    auto const& delta  = p.opt<double>(
        "delta", 25.0, "Gaussian penalty width δ in (S−E_n)²/2δ² (also the a-update scale)");
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
    if (!p.parse(argc, argv)) {
        return 0;
    }

    log::start(cf.workspace, cf.out, /*replicas=*/true);
    std::string const outpath = app::out_path(cf);

    // ---- Base action ----
    Lattice<double>::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(cf.L));
    act::Phi4<double> const base{.kappa = kappa, .lambda = lambda};
    log::act(base);

    // ---- Replica geometry ----
    double const d_e = spacing > 0.0 ? spacing : delta;
    int const n_rep  = std::max(2, static_cast<int>(std::lround((e_max - e_min) / d_e)) + 1);
    double const e_max_snapped = e_min + (static_cast<double>(n_rep - 1) * d_e);

    // ---- Replicas (cold config; warmed into their window below) ----
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

    log::info("llr", "warm {} replicas into window", n_rep);
    cuda::llr::warm_all(reps, 50, 10, 1.0);

    // ---- Output ----
    FastRng exch_rng{cf.seed};
    io::Writer out{outpath, argc, argv, &p};
    out.start_phase("llr");

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
