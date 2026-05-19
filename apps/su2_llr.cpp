// LLR (Gaussian-penalty) with replica exchange for SU(2) Wilson gauge theory.
//
// Mirrors u1_llr.cpp but on `MatrixLinkLattice<SU2, double>` with the
// generic `Wilson<SU2>` action. Each replica is cold-initialised to the SU(2)
// identity (all matrices = I) since the default field ctor leaves zeros,
// which would be an invalid group element.
//
// Energy variable: E(U) = S_W(U) (full Wilson action; bounded below by 0).
// Sampler:        SU(2) HMC with templated integrator (default Omelyan2).
// Update:         Newton-Raphson warm-up then restarted Robbins-Monro.
// Geometry:       n_rep replicas at E_n = E_min + n * delta.
// Exchange:       even/odd alternating nearest-neighbour swaps.

#include <reticolo/reticolo.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    using namespace reticolo;
    using Group    = gauge_group::SU2;
    using Action   = action::Wilson<Group, double>;
    using Field    = MatrixLinkLattice<Group, double>;
    using ReplicaT = llr::Replica<Action, FastRng, alg::integ::Omelyan2, double, Field>;

    cli::Parser p{"su2_llr", "LLR with replica exchange for SU(2) Wilson action"};
    auto const& L     = p.req<int>("L,size", "linear lattice extent");
    auto const& ndim  = p.opt<int>("ndim", 4, "spatial dimensions");
    auto const& beta  = p.req<double>("beta", "Wilson coupling");
    auto const& e_min = p.req<double>("E_min", "lower window centre");
    auto const& e_max = p.req<double>("E_max", "upper window centre");
    auto const& delta =
        p.req<double>("delta", "single LLR tuning knob: Gaussian half-width AND replica spacing.");
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
    auto const& outpath   = p.opt<std::string>("out", std::string{"su2_llr.h5"}, "HDF5 output");
    if (!p.parse(argc, argv)) {
        return 0;
    }

    Field::SizeVec shape(static_cast<std::size_t>(ndim), static_cast<std::size_t>(L));
    Action const base{.beta = beta};

    int const n_rep  = std::max(2, static_cast<int>(std::lround((e_max - e_min) / delta)) + 1);
    double const d_e = delta;
    double const e_max_snapped = e_min + (static_cast<double>(n_rep - 1) * d_e);
    double const a_init        = 0.0;

    std::vector<std::unique_ptr<ReplicaT>> reps;
    reps.reserve(static_cast<std::size_t>(n_rep));
    for (int n = 0; n < n_rep; ++n) {
        double const e_n = e_min + (static_cast<double>(n) * d_e);
        reps.push_back(
            std::make_unique<ReplicaT>(shape,
                                       base,
                                       FastRng{seed + 1ULL + static_cast<unsigned long long>(n)},
                                       e_n,
                                       delta,
                                       a_init,
                                       alg::HmcSpec{.tau = tau, .n_md = n_md}));
        // Cold-start each replica's field to SU(2) identity (Re U_{00} =
        // Re U_{11} = 1, all else 0).
        Field& phi           = reps.back()->phi();
        std::size_t const ns = phi.nsites();
        for (std::size_t mu = 0; mu < static_cast<std::size_t>(ndim); ++mu) {
            double* const blk = phi.mu_block_data(mu);
            for (std::size_t s = 0; s < ns; ++s) {
                blk[(0 * ns) + s] = 1.0;
                blk[(6 * ns) + s] = 1.0;
            }
        }
    }

    FastRng exch_rng{seed};
    io::Writer out{outpath, argc, argv, &p};
    out.start_phase("llr");

    out.attr<int>("/cfg@n_rep", n_rep);
    out.attr<int>("/cfg@n_nr", n_nr);
    out.attr<int>("/cfg@n_rm", n_rm);
    out.attr<double>("/cfg@delta", delta);
    out.attr<double>("/cfg@E_min", e_min);
    out.attr<double>("/cfg@E_max", e_max_snapped);
    out.attr<double>("/cfg@dE", d_e);

    auto e_n_series = out.series<double>("/cfg/E_n");
    for (auto const& r : reps) {
        e_n_series.append(r->E_n());
    }

    std::vector<io::Series<double>> a_series;
    std::vector<io::Series<double>> de_series;
    a_series.reserve(static_cast<std::size_t>(n_rep));
    de_series.reserve(static_cast<std::size_t>(n_rep));
    for (int n = 0; n < n_rep; ++n) {
        a_series.emplace_back(out.series<double>(std::format("/replica_{:03d}/a", n)));
        de_series.emplace_back(out.series<double>(std::format("/replica_{:03d}/dE", n)));
    }
    auto exch_series = out.series<int>("/exchange/accepted");

    std::size_t const n_rep_u = static_cast<std::size_t>(n_rep);
    std::vector<double> de_buf(n_rep_u);
    std::vector<double> a_buf(n_rep_u);

    // Newton-Raphson warm-up.
    for (int k = 0; k < n_nr; ++k) {
#pragma omp parallel for schedule(dynamic, 1)
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            auto& r = *reps[n];
            r.thermalize(n_therm_nr);
            de_buf[n] = r.sample(n_meas_nr);
            a_buf[n]  = llr::nr_update(r.a(), de_buf[n], delta);
            r.set_a(a_buf[n]);
        }
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            a_series[n].append(a_buf[n]);
            de_series[n].append(de_buf[n]);
        }
    }

    // Robbins-Monro with replica exchange.
    for (int s = 0; s < n_rm; ++s) {
#pragma omp parallel for schedule(dynamic, 1)
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            auto& r = *reps[n];
            r.thermalize(n_therm_rm);
            de_buf[n] = r.sample(n_meas_rm);
            a_buf[n]  = llr::rm_update(r.a(), de_buf[n], delta, s);
            r.set_a(a_buf[n]);
        }
        for (std::size_t n = 0; n < n_rep_u; ++n) {
            a_series[n].append(a_buf[n]);
            de_series[n].append(de_buf[n]);
        }

        std::size_t const off = static_cast<std::size_t>(s & 1);
        int accepted          = 0;
        for (std::size_t i = off; i + 1 < reps.size(); i += 2) {
            if (llr::try_exchange(*reps[i], *reps[i + 1], exch_rng)) {
                ++accepted;
            }
        }
        exch_series.append(accepted);
    }
}
