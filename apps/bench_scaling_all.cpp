#include <reticolo/reticolo.hpp>

#include "_bench/hot_init.hpp"

#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string_view>

// Cross-family OpenMP scaling bench: for one lattice size L (argv[1]) and the
// ambient OMP_NUM_THREADS, time compute_force / s_full / one HMC trajectory for
// every action family (site, bond, complex, gauge). A driver sweeps threads × L
// and turns the per-op wall times into speedups. Silent + hot-initialised so the
// timed loop measures real kernel arithmetic, not I/O or zero-folds.

using namespace reticolo;
using clk = std::chrono::steady_clock;

template <class F>
double time_ms(F&& f, int reps) {
    for (int i = 0; i < 3; ++i) {
        f();
    }
    auto const t0 = clk::now();
    for (int i = 0; i < reps; ++i) {
        f();
    }
    return std::chrono::duration<double>(clk::now() - t0).count() / reps * 1e3;
}

int g_rf = 20;  // force/s_full timed reps
int g_rt = 8;   // trajectory timed reps

template <class A, class Field>
void bench_one(char const* name, A const& a, Field& fld, FastRng& rng, int L, char const* th) {
    Field force{fld.indexing()};
    double const fms = time_ms([&] { a.compute_force(fld, force); }, g_rf);
    double sink      = 0.0;
    double const sms = time_ms([&] { sink += a.s_full(fld); }, g_rf);
    double trms      = 0.0;  // g_rt <= 0 skips the (expensive) trajectory
    if (g_rt > 0) {
        alg::Hmc hmc{a, fld, rng, {.tau = 1.0, .n_md = 8}, alg::integ::leapfrog, log::Mode::silent};
        trms = time_ms([&] { (void)hmc.step(log::Mode::silent); }, g_rt);
    }
    std::printf("%-11s %-3d %-2s %10.4f %10.4f %10.4f\n", name, L, th, fms, sms, trms);
    (void)sink;
}

// Usage: bench_scaling_all <L> [family=all|light|gauge] [force_reps] [traj_reps].
// `light` = site+bond+complex (8-16 B/site) so L can grow far past cache without
// the gauge OOM/blow-up.
int main(int argc, char** argv) {
    reticolo::log::off();
    int const L      = argc > 1 ? std::atoi(argv[1]) : 24;
    std::string_view const fam = argc > 2 ? argv[2] : "all";
    g_rf                       = argc > 3 ? std::atoi(argv[3]) : 20;
    g_rt                       = argc > 4 ? std::atoi(argv[4]) : 8;
    bool const phi_only        = (fam == "phi");           // Phi4 + Phi6 only
    bool const light           = phi_only || (fam != "gauge");
    bool const gauge           = (fam == "all" || fam == "gauge");
    char const* th   = std::getenv("OMP_NUM_THREADS");
    if (th == nullptr) {
        th = "1";
    }
    auto const n = static_cast<std::size_t>(L);
    FastRng rng{42};

    if (light) {
        // ---- site + bond (Lattice<double>) ----
        {
            Lattice<double> phi{Lattice<double>::SizeVec(4, n)};
            reticolo::bench::hot_init(phi, rng);
            bench_one("Phi4", act::Phi4<double>{.kappa = 0.18, .lambda = 1.0}, phi, rng, L, th);
            bench_one(
                "Phi6", act::Phi6<double>{.kappa = 0.18, .lambda = 1.0, .g6 = 0.1}, phi, rng, L, th);
            if (!phi_only) {
                bench_one("SineGordon",
                          act::SineGordon<double>{.kappa = 0.18, .alpha = 0.7},
                          phi,
                          rng,
                          L,
                          th);
                bench_one("Xy", act::Xy<double>{.beta = 1.0}, phi, rng, L, th);
            }
        }
        // ---- complex (Lattice<std::complex<double>>) ----
        if (!phi_only) {
            using CF = Lattice<std::complex<double>>;
            CF phi{CF::SizeVec(4, n)};
            reticolo::bench::hot_init(phi, rng);
            bench_one("BoseGas",
                      act::BoseGas<double>{.mass = 1.0, .lambda = 1.0, .mu = 0.5},
                      phi,
                      rng,
                      L,
                      th);
        }
    }
    if (gauge) {
        {
            using F = MatrixLinkLattice<gauge_group::U1, double>;
            F u{F::SizeVec(4, n)};
            reticolo::bench::hot_init(u, rng);
            bench_one("Wilson<U1>", act::Wilson<gauge_group::U1, double>{.beta = 1.0}, u, rng, L, th);
        }
        {
            using F = MatrixLinkLattice<gauge_group::SU2, double>;
            F u{F::SizeVec(4, n)};
            reticolo::bench::hot_init(u, rng);
            bench_one(
                "Wilson<SU2>", act::Wilson<gauge_group::SU2, double>{.beta = 2.3}, u, rng, L, th);
        }
        {
            using F = MatrixLinkLattice<gauge_group::SU3, double>;
            F u{F::SizeVec(4, n)};
            reticolo::bench::hot_init(u, rng);
            bench_one(
                "Wilson<SU3>", act::Wilson<gauge_group::SU3, double>{.beta = 6.0}, u, rng, L, th);
        }
    }
    return 0;
}
