#include <reticolo/reticolo.hpp>

#include "_bench/hot_init.hpp"

#include <algorithm>
#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

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
void bench_one(
    char const* name, A const& a, Field& fld, FastRng& rng, char const* shape, char const* th) {
    Field force{fld.indexing()};
    double const fms = time_ms([&] { a.compute_force(fld, force); }, g_rf);
    double sink      = 0.0;
    double const sms = time_ms([&] { sink += a.s_full(fld); }, g_rf);
    double trms      = 0.0;  // g_rt <= 0 skips the (expensive) trajectory
    if (g_rt > 0) {
        alg::Hmc hmc{a, fld, rng, {.tau = 1.0, .n_md = 8}, alg::integ::leapfrog, log::Mode::silent};
        trms = time_ms([&] { (void)hmc.step(log::Mode::silent); }, g_rt);
    }
    // Working-set MB of one field pass and the thread count the policy actually
    // picks for it at the ambient OMP ceiling (traverse_threads) — the spawn
    // decision, so the sweep shows where threading kicks in per family.
    std::size_t const bps = fld.bytes_per_site();
    std::size_t const n   = fld.nsites();
    double const mb       = static_cast<double>(n * bps) / (1024.0 * 1024.0);
    int const nthr        = std::min<int>(reticolo::exec::traverse_threads(n, bps),
                                          static_cast<int>(reticolo::exec::partition(fld).n_items));
    std::printf("%-11s %-13s %-3s %10.4f %10.4f %10.4f %9.2f %5d\n",
                name,
                shape,
                th,
                fms,
                sms,
                trms,
                mb,
                nthr);
    (void)sink;
}

// argv[1] is either a cubic edge ("24" → 4D 24⁴) or an explicit "AxBxCxD" shape
// (weak scaling grows the outermost/last dim). Returns the site-range shape.
std::vector<std::size_t> parse_shape(char const* s) {
    std::vector<std::size_t> dims;
    std::size_t v = 0;
    bool any      = false;
    for (char const* p = s;; ++p) {
        if (*p >= '0' && *p <= '9') {
            v   = v * 10 + static_cast<std::size_t>(*p - '0');
            any = true;
        } else {
            if (any) {
                dims.push_back(v);
            }
            v   = 0;
            any = false;
            if (*p == 0) {
                break;
            }
        }
    }
    if (dims.size() == 1) {
        return std::vector<std::size_t>(4, dims[0]);  // cubic 4D
    }
    return dims;
}

// Usage: bench_scaling_all <shape> [family=all|light|gauge|phi|pair] [force_reps] [traj_reps].
// <shape> is a cubic edge ("24" → 24⁴) or explicit "AxBxCxD". `light` =
// site+bond+complex (8-16 B/site) so the volume can grow past cache without the
// gauge OOM/blow-up; `phi` = Phi4+Phi6; `pair` = Phi4 + Wilson<SU3> (the two
// scaling regimes: memory-bound site vs compute-bound gauge).
int main(int argc, char** argv) {
    reticolo::log::off();
    char const* const shape_arg = argc > 1 ? argv[1] : "24";
    std::string_view const fam  = argc > 2 ? argv[2] : "all";
    g_rf                        = argc > 3 ? std::atoi(argv[3]) : 20;
    g_rt                        = argc > 4 ? std::atoi(argv[4]) : 8;
    bool const pair             = (fam == "pair");         // Phi4 + Wilson<SU3>
    bool const phi_only         = (fam == "phi") || pair;  // no bond/complex extras
    bool const light            = phi_only || (fam != "gauge");
    bool const gauge            = (fam == "all" || fam == "gauge" || pair);
    char const* th              = std::getenv("OMP_NUM_THREADS");
    if (th == nullptr) {
        th = "1";
    }
    std::vector<std::size_t> const shape = parse_shape(shape_arg);
    FastRng rng{42};

    if (light) {
        // ---- site + bond (Lattice<double>) ----
        {
            Lattice<double> phi{shape};
            reticolo::bench::hot_init(phi, rng);
            bench_one(
                "Phi4", act::Phi4<double>{.kappa = 0.18, .lambda = 1.0}, phi, rng, shape_arg, th);
            if (!pair) {
                bench_one("Phi6",
                          act::Phi6<double>{.kappa = 0.18, .lambda = 1.0, .g6 = 0.1},
                          phi,
                          rng,
                          shape_arg,
                          th);
            }
            if (!phi_only) {
                bench_one("SineGordon",
                          act::SineGordon<double>{.kappa = 0.18, .alpha = 0.7},
                          phi,
                          rng,
                          shape_arg,
                          th);
                bench_one("Xy", act::Xy<double>{.beta = 1.0}, phi, rng, shape_arg, th);
            }
        }
        // ---- complex (Lattice<std::complex<double>>) ----
        if (!phi_only) {
            using CF = Lattice<std::complex<double>>;
            CF phi{shape};
            reticolo::bench::hot_init(phi, rng);
            bench_one("BoseGas",
                      act::BoseGas<double>{.mass = 1.0, .lambda = 1.0, .mu = 0.5},
                      phi,
                      rng,
                      shape_arg,
                      th);
        }
    }
    if (gauge) {
        if (!pair) {
            {
                using F = MatrixLinkLattice<math::group::U1, double>;
                F u{shape};
                reticolo::bench::hot_init(u, rng);
                bench_one("Wilson<U1>",
                          act::Wilson<math::group::U1, double>{.beta = 1.0},
                          u,
                          rng,
                          shape_arg,
                          th);
            }
            {
                using F = MatrixLinkLattice<math::group::SU2, double>;
                F u{shape};
                reticolo::bench::hot_init(u, rng);
                bench_one("Wilson<SU2>",
                          act::Wilson<math::group::SU2, double>{.beta = 2.3},
                          u,
                          rng,
                          shape_arg,
                          th);
            }
        }
        {
            using F = MatrixLinkLattice<math::group::SU3, double>;
            F u{shape};
            reticolo::bench::hot_init(u, rng);
            bench_one("Wilson<SU3>",
                      act::Wilson<math::group::SU3, double>{.beta = 6.0},
                      u,
                      rng,
                      shape_arg,
                      th);
        }
    }
    return 0;
}
