// Head-to-head perf bench: CompactU1 (hand-tuned, Sleef batched cos/sin path)
// vs Wilson<U1> (generic concept-driven path). Same lattice, same seed, same
// integrator settings — the only difference is the action template and the
// underlying field type (LinkLattice<double> vs MatrixLinkLattice<U1, double>,
// which have identical storage at nc = 1).
//
// Decides M8: if Wilson<U1> is within ~5% of CompactU1 we can route the U(1)
// apps through Wilson<U1> and delete compact_u1.hpp. Otherwise keep both.

#include <reticolo/reticolo.hpp>

#include <chrono>
#include <cstddef>
#include <cstdio>

namespace {

using bench_clock = std::chrono::steady_clock;

double seconds(bench_clock::duration d) {
    return std::chrono::duration<double>(d).count();
}

}  // namespace

int main() {
    using namespace reticolo;
    log::off();
    using Integ = alg::integ::Omelyan2;

    constexpr int k_n_md    = 20;
    constexpr double k_tau  = 1.0;
    constexpr double k_beta = 1.01;
    constexpr int k_warmup  = 30;
    constexpr int k_n_traj  = 200;

    struct Case {
        int ndim;
        int L;
    };
    Case const cases[] = {
        {.ndim = 3, .L = 12},
        {.ndim = 4, .L = 6},
        {.ndim = 4, .L = 8},
    };

    std::printf(
        "%-12s %-12s %-14s %-14s %-10s\n", "ndim x L", "action", "traj [s]", "dof upd/s", "accept");
    std::printf(
        "%-12s %-12s %-14s %-14s %-10s\n", "--------", "------", "--------", "---------", "------");

    for (auto const& c : cases) {
        std::size_t const nd = static_cast<std::size_t>(c.ndim);
        std::size_t const L_ = static_cast<std::size_t>(c.L);

        // ---- CompactU1 (LinkLattice<double>) ----
        double compact_seconds_per_traj = 0.0;
        double compact_accept           = 0.0;
        {
            using Action = action::CompactU1<double>;
            LinkLattice<double>::SizeVec shape(nd, L_);
            LinkLattice<double> theta{shape, 0.0};
            FastRng rng{42};
            Action const action{.beta = k_beta};
            alg::Hmc<Action, FastRng, Integ, LinkLattice<double>> hmc{
                action, theta, rng, {.tau = k_tau, .n_md = k_n_md}};
            for (int i = 0; i < k_warmup; ++i) {
                (void)hmc.trajectory();
            }
            int acc       = 0;
            auto const t0 = bench_clock::now();
            for (int i = 0; i < k_n_traj; ++i) {
                acc += hmc.trajectory().accepted ? 1 : 0;
            }
            compact_seconds_per_traj = seconds(bench_clock::now() - t0) / k_n_traj;
            compact_accept           = static_cast<double>(acc) / k_n_traj;
        }

        // ---- Wilson<U1> (MatrixLinkLattice<U1, double>) ----
        double wilson_seconds_per_traj = 0.0;
        double wilson_accept           = 0.0;
        {
            using Group  = gauge_group::U1;
            using Action = action::Wilson<Group, double>;
            using Field  = MatrixLinkLattice<Group, double>;
            Field::SizeVec shape(nd, L_);
            Field theta{shape};
            FastRng rng{42};
            Action const action{.beta = k_beta};
            alg::Hmc<Action, FastRng, Integ, Field> hmc{
                action, theta, rng, {.tau = k_tau, .n_md = k_n_md}};
            for (int i = 0; i < k_warmup; ++i) {
                (void)hmc.trajectory();
            }
            int acc       = 0;
            auto const t0 = bench_clock::now();
            for (int i = 0; i < k_n_traj; ++i) {
                acc += hmc.trajectory().accepted ? 1 : 0;
            }
            wilson_seconds_per_traj = seconds(bench_clock::now() - t0) / k_n_traj;
            wilson_accept           = static_cast<double>(acc) / k_n_traj;
        }

        std::size_t nsites = L_;
        for (std::size_t i = 1; i < nd; ++i) {
            nsites *= L_;
        }
        std::size_t const dofs = nd * nsites;
        double const compact_dof_per_s =
            static_cast<double>(dofs * k_n_md) / compact_seconds_per_traj;
        double const wilson_dof_per_s =
            static_cast<double>(dofs * k_n_md) / wilson_seconds_per_traj;

        std::printf("%dD L=%-2d     %-12s %-14.3e %-14.3e %-10.3f\n",
                    c.ndim,
                    c.L,
                    "CompactU1",
                    compact_seconds_per_traj,
                    compact_dof_per_s,
                    compact_accept);
        std::printf("%dD L=%-2d     %-12s %-14.3e %-14.3e %-10.3f\n",
                    c.ndim,
                    c.L,
                    "Wilson<U1>",
                    wilson_seconds_per_traj,
                    wilson_dof_per_s,
                    wilson_accept);
        std::printf("            ratio (Wilson/CompactU1) = %.3f\n\n",
                    wilson_seconds_per_traj / compact_seconds_per_traj);
    }
}
