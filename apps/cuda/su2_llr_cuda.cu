// LLR (Gaussian-penalty) with replica exchange for SU(2) Wilson gauge theory on
// the CUDA backend. The GPU twin of su2_llr.cpp: same CLI + LLR schedule, same
// param-swap output schema as phi4_llr_cuda / u1_llr_cuda. The gauge windowed
// action is the generic device WindowedAction over a MatrixLayout<SU2> field
// (two-pass; no fused s_full_and_force yet).
//
// Cold start = SU(2) identity via the trait's set_cold (the zero the default
// buffer holds is not a group element). No hot-start: β≈2.3 is smooth confined
// (no bulk transition), so the NR warm-in reaches the windows from cold — same as
// the CPU su2_llr.
//
// Output schema (HDF5): identical to phi4_llr_cuda — /cfg@*, /cfg/E_n,
// /replica_NNN/{a,dE,E_n}, /exchange/accepted. Group by E_n downstream.

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
    using Group    = math::group::SU2;
    using Action   = action::Wilson<Group, double>;
    using DField   = cuda::DeviceField<double, cuda::MatrixLayout<Group>>;
    using ReplicaT = cuda::llr::Replica<Action, updater::integ::Omelyan2, DField>;

    // ---- CLI (mirrors su2_llr.cpp) ----
    cli::Parser p{"su2_llr_cuda",
                  "LLR (Gaussian-penalty) + replica exchange for SU(2) Wilson (CUDA)"};
    auto const cf     = app::common_flags(p, {.L = 4, .out = "su2_llr_cuda.h5"});
    auto const& ndim  = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& beta  = p.opt<double>("beta", 2.3, "Wilson coupling");
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
    if (!p.parse(argc, argv)) {
        return 0;
    }

    log::start(cf.workspace, cf.out, /*replicas=*/true);
    std::string const outpath = app::out_path(cf);

    // ---- Base action ----
    std::vector<std::size_t> const shape(static_cast<std::size_t>(ndim),
                                         static_cast<std::size_t>(cf.L));
    Action const base{.beta = beta};
    log::act(base);

    // ---- Replica geometry ----
    double const d_e = spacing > 0.0 ? spacing : delta;
    int const n_rep  = std::max(2, static_cast<int>(std::lround((e_max - e_min) / d_e)) + 1);
    double const e_max_snapped = e_min + (static_cast<double>(n_rep - 1) * d_e);

    // ---- Replicas (cold = SU(2) identity via set_cold; warmed into window) ----
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
