// Smoothed LLR variant for compact U(1): per-replica Robbins-Monro with a
// cross-replica local-quadratic smoother shrunk into each iterate. See
// include/reticolo/llr/smoothed_driver.hpp.
//
// Mirrors apps/u1_llr.cpp: same action, sampler, geometry, exchange and
// HDF5 schema; the only differences are the driver call and four extra
// CLI knobs controlling the smoother and shrinkage schedule.

#include <reticolo/llr/smoothed_driver.hpp>
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

    cli::Parser p{"u1_llr_smoothed", "Smoothed LLR for compact U(1) Wilson action"};
    auto const cf     = app::common_flags(p, {.L = 4, .out = "u1_llr_smoothed.h5"});
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
    auto const& smooth_K =
        p.opt<int>("smooth_K", 4, "neighbours each side in the local-polynomial fit");
    auto const& smooth_degree =
        p.opt<int>("smooth_degree", 2, "polynomial degree of the smoother (>= 1)");
    auto const& smooth_lambda0 = p.opt<double>(
        "smooth_lambda0", 1.0, "shrinkage weight prefactor (lambda_s = lambda0/(s+1)^exp)");
    auto const& smooth_lambda_exp = p.opt<double>(
        "smooth_lambda_exp", 2.0, "shrinkage decay exponent; >1 keeps the perturbation summable");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    log::start(cf.workspace, cf.out, /*replicas=*/true);
    std::string const outpath = app::out_path(cf);

    MatrixLinkLattice<math::group::U1, double>::SizeVec shape(static_cast<std::size_t>(ndim),
                                                              static_cast<std::size_t>(cf.L));
    Action const base{.beta = beta};
    log::act(base);

    double const d_e = spacing > 0.0 ? spacing : delta;
    int const n_rep  = std::max(2, static_cast<int>(std::lround((e_max - e_min) / d_e)) + 1);
    double const e_max_snapped = e_min + (static_cast<double>(n_rep - 1) * d_e);

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

    FastRng exch_rng{cf.seed};
    io::Writer out{outpath, argc, argv, &p};
    out.start_phase("llr");

    constexpr double k_hot_sigma   = std::numbers::pi;
    constexpr int k_warm_batches   = 50;
    constexpr int k_warm_batch_len = 10;
    std::size_t const n_rep_u      = static_cast<std::size_t>(n_rep);
#pragma omp parallel for schedule(dynamic, 1)
    for (std::size_t n = 0; n < n_rep_u; ++n) {
        auto _ = log::scope(reps[n]->id());
        reps[n]->hot_start(k_hot_sigma);
        reps[n]->warm_into_window(k_warm_batches, k_warm_batch_len, 1.0);
    }

    llr::smoothed::run(reps,
                       exch_rng,
                       llr::smoothed::DriverSpec{.n_nr              = n_nr,
                                                 .n_therm_nr        = n_therm_nr,
                                                 .n_meas_nr         = n_meas_nr,
                                                 .n_rm              = n_rm,
                                                 .n_therm_rm        = n_therm_rm,
                                                 .n_meas_rm         = n_meas_rm,
                                                 .delta             = delta,
                                                 .e_min             = e_min,
                                                 .E_max             = e_max_snapped,
                                                 .d_e               = d_e,
                                                 .exchange          = (exchange != 0),
                                                 .smooth_K          = smooth_K,
                                                 .smooth_degree     = smooth_degree,
                                                 .smooth_lambda0    = smooth_lambda0,
                                                 .smooth_lambda_exp = smooth_lambda_exp},
                       out);
}
