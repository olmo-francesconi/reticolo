// bench_integrator_efficiency — the HMC integrator trade-off, measured.
//
// The library makes the MD integrator a swappable type parameter, so the three
// integrators can be compared on identical footing. Higher order costs more
// force evaluations per step but the energy violation falls off faster with the
// step size (dt^2 for Leapfrog / Omelyan2, dt^4 for Omelyan4), so a coarser
// step suffices at the same accuracy. The fair cost axis is therefore force
// evaluations per trajectory, and the accuracy axis is sigma(dH) (the width of
// the energy violation, which sets the acceptance via <P_acc> = erfc(sqrt(<dH>/2))
// and, unlike acceptance, does not saturate at 1).
//
// For each action we thermalise once to equilibrium with a reliable integrator,
// then for every (integrator, n_md) measure acceptance, <dH> and sigma(dH) on a
// copy of that equilibrium configuration. phi^4 (scalar) and Wilson<SU3> (gauge)
// are run so the scalar and gauge behaviour can be compared.

#include <reticolo/reticolo.hpp>

#include "_bench/hot_init.hpp"
#include "_bench/timing.hpp"

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace {

using namespace reticolo;

struct Result {
    int n_md;
    int force_evals;
    double wall;     // s / trajectory
    double acc;
    double dh_mean;
    double dh_std;
};

template <class Field, class Action, class Integ>
Result measure(Field const& thermalised, Action const& action, Integ integ,
               int force_per_step, int n_md, double tau, int n_meas) {
    Field phi = thermalised;  // start each point from the same equilibrium config
    FastRng rng{9001};
    alg::Hmc hmc{action, phi, rng, {.tau = tau, .n_md = n_md}, integ, log::Mode::silent};

    long accepted = 0;
    double sum_dh = 0.0, sum_dh2 = 0.0;
    for (int m = 0; m < n_meas; ++m) {
        auto const st = hmc.step(log::Mode::silent);
        accepted += st.accepted ? 1 : 0;
        sum_dh += st.dH;
        sum_dh2 += st.dH * st.dH;
    }
    double const n  = n_meas;
    double const mu = sum_dh / n;
    double const var = (sum_dh2 / n) - (mu * mu);
    double const wall = bench::time_per_call([&] { (void)hmc.step(log::Mode::silent); });
    return {.n_md        = n_md,
            .force_evals = (force_per_step * n_md) + 1,
            .wall        = wall,
            .acc         = static_cast<double>(accepted) / n,
            .dh_mean     = mu,
            .dh_std      = std::sqrt(var > 0.0 ? var : 0.0)};
}

struct Sweep {
    char const* name;
    int force_per_step;
    std::vector<int> n_mds;
};

template <class Field, class Action>
void run_action(char const* label, Field& phi, Action const& action, double tau, int n_meas) {
    using namespace reticolo::alg::integ;

    // Thermalise once with a reliable integrator; every point starts from a
    // copy of this equilibrium config (HMC targets exp(-S) for all integrators).
    FastRng warm_rng{42};
    {
        alg::Hmc warm{action, phi, warm_rng, {.tau = tau, .n_md = 24}, omelyan2, log::Mode::silent};
        for (int i = 0; i < 400; ++i) {
            (void)warm.step(log::Mode::silent);
        }
    }

    std::printf("=== %s (tau=%.2f) ===\n", label, tau);
    std::printf("%-10s %-6s %-12s %-11s %-9s %-11s %-11s\n",
                "integrator", "n_md", "force_evals", "wall[s]", "acc", "<dH>", "sigma(dH)");

    Sweep const sweeps[] = {
        {"Leapfrog", 1, {8, 12, 16, 24, 32, 48, 64}},
        {"Omelyan2", 2, {4, 6, 8, 12, 16, 24, 32}},
        {"Omelyan4", 4, {2, 3, 4, 6, 8, 12, 16}},
    };
    for (auto const& s : sweeps) {
        for (int n_md : s.n_mds) {
            Result r{};
            if (s.force_per_step == 1) {
                r = measure(phi, action, leapfrog, 1, n_md, tau, n_meas);
            } else if (s.force_per_step == 2) {
                r = measure(phi, action, omelyan2, 2, n_md, tau, n_meas);
            } else {
                r = measure(phi, action, omelyan4, 4, n_md, tau, n_meas);
            }
            std::printf("%-10s %-6d %-12d %-11.3e %-9.4f %-11.3e %-11.3e\n",
                        s.name, r.n_md, r.force_evals, r.wall, r.acc, r.dh_mean, r.dh_std);
        }
    }
    std::printf("\n");
}

}  // namespace

int main() {
    reticolo::log::off();
    std::printf("INTEGRATOR EFFICIENCY — acceptance / energy violation vs cost\n"
                "(thermalised ensemble; higher order = smaller sigma(dH) at fixed step)\n\n");

    {
        Lattice<double>::SizeVec shape(4, 8);
        Lattice<double> phi{shape};
        FastRng seed{7};
        bench::hot_init(phi, seed);
        act::Phi4<double> const action{.kappa = 0.12, .lambda = 1.0};
        run_action("phi4  4D L=8  kappa=0.12", phi, action, 1.0, 400);
    }
    {
        using Field = MatrixLinkLattice<gauge_group::SU3, double>;
        Field::SizeVec shape(4, 6);
        Field u{shape};
        FastRng seed{7};
        bench::hot_init(u, seed);
        act::Wilson<gauge_group::SU3, double> const action{.beta = 6.0};
        run_action("Wilson<SU3>  4D L=6  beta=6.0", u, action, 1.0, 300);
    }
}
