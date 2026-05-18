// Wall-time benchmark for all scalar HMC-capable actions (Phi4, Phi6, SineGordon).
// Same harness as bench_phi4, parameterised on the action type.

#include <reticolo/reticolo.hpp>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

namespace {

using bench_clock = std::chrono::steady_clock;

double seconds(bench_clock::duration d) {
    return std::chrono::duration<double>(d).count();
}

struct Case {
    int ndim;
    int L;
    int n_mc;
    int n_hmc;
};

template <class Action>
void run_one(std::string const& name, Action const& action, std::vector<Case> const& cases,
             double sigma, int n_md) {
    std::printf("\n=== %s ===\n", name.c_str());
    std::printf("%-12s %-10s   %-14s %-18s   %-14s %-18s %-12s\n",
                "ndim x L", "V", "MC sweep [s]", "MC throughput",
                "HMC traj [s]", "HMC throughput", "HMC accept");
    std::printf("%-12s %-10s   %-14s %-18s   %-14s %-18s %-12s\n",
                "--------", "-", "------------", "-------------",
                "------------", "--------------", "----------");

    using namespace reticolo;
    for (auto const& c : cases) {
        Lattice<double>::SizeVec shape(static_cast<std::size_t>(c.ndim),
                                       static_cast<std::size_t>(c.L));
        Lattice<double>          phi{shape};
        FastRng                  rng{42};
        auto const               nsites = phi.nsites();

        // Warmup.
        alg::Hmc<Action, FastRng> warm{action, phi, rng, {.tau = 1.0, .n_md = n_md}};
        for (int i = 0; i < 50; ++i) {
            (void)warm.trajectory();
        }

        // MC.
        alg::Metropolis<Action, FastRng> mc{action, phi, rng, sigma};
        auto const                       t0 = bench_clock::now();
        for (int i = 0; i < c.n_mc; ++i) {
            (void)mc.sweep();
        }
        double const mc_per     = seconds(bench_clock::now() - t0) / c.n_mc;
        double const mc_thru    = static_cast<double>(nsites) / mc_per;

        // HMC.
        alg::Hmc<Action, FastRng> hmc{action, phi, rng, {.tau = 1.0, .n_md = n_md}};
        int                       accepted = 0;
        auto const                t1       = bench_clock::now();
        for (int i = 0; i < c.n_hmc; ++i) {
            accepted += hmc.trajectory().accepted ? 1 : 0;
        }
        double const h_per    = seconds(bench_clock::now() - t1) / c.n_hmc;
        double const h_thru   = (static_cast<double>(nsites) * static_cast<double>(n_md)) / h_per;
        double const accept   = static_cast<double>(accepted) / c.n_hmc;

        std::printf("%-2dD x L=%-4d %-10zu   %-13.3e %-12.1f M/s   %-13.3e %-12.1f M/s %-10.3f\n",
                    c.ndim, c.L, nsites,
                    mc_per, mc_thru / 1e6, h_per, h_thru / 1e6, accept);
    }
}

}  // namespace

int main() {
    using namespace reticolo;

    std::vector<Case> const cases = {
        {.ndim = 3, .L = 8,  .n_mc = 500, .n_hmc = 200},
        {.ndim = 3, .L = 16, .n_mc = 200, .n_hmc = 100},
        {.ndim = 3, .L = 24, .n_mc = 100, .n_hmc = 50},
        {.ndim = 4, .L = 6,  .n_mc = 500, .n_hmc = 200},
        {.ndim = 4, .L = 10, .n_mc = 100, .n_hmc = 50},
    };
    constexpr int k_n_md = 20;

    run_one("Phi4",        act::Phi4<double>{.kappa = 0.18, .lambda = 1.145},
            cases, /*sigma=*/0.4, k_n_md);
    run_one("Phi6 (g6=0)", act::Phi6<double>{.kappa = 0.18, .lambda = 1.145, .g6 = 0.0},
            cases, /*sigma=*/0.4, k_n_md);
    run_one("Phi6 (g6=0.2)", act::Phi6<double>{.kappa = 0.18, .lambda = 1.145, .g6 = 0.2},
            cases, /*sigma=*/0.4, k_n_md);
    run_one("SineGordon",  act::SineGordon<double>{.kappa = 0.18, .alpha = 1.0},
            cases, /*sigma=*/0.4, k_n_md);
}
