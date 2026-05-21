// HMC throughput: phase-quenched 4D Bose gas (complex field) vs reference
// real-scalar Phi4 (real field). "Phase-quenched" here just means HMC
// samples exp(−S_R) where S_R is the real part of the complex Bose-gas
// action — there is no algorithmic overhead beyond storing complex fields
// and complex momenta vs real ones. Each complex site carries 2 real DOFs,
// so the fair throughput metric is "real-DOF updates per second" and we
// report it for both actions.

#include <reticolo/reticolo.hpp>

#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace {

using bench_clock = std::chrono::steady_clock;

double seconds(bench_clock::duration d) {
    return std::chrono::duration<double>(d).count();
}

struct Case {
    int ndim;
    int L;
    int n_traj;
};

}  // namespace

int main() {
    using namespace reticolo;
    log::off();
    using Integ = alg::integ::Omelyan2;

    std::vector<Case> const cases = {
        {.ndim = 3, .L = 8, .n_traj = 400},
        {.ndim = 3, .L = 12, .n_traj = 200},
        {.ndim = 3, .L = 16, .n_traj = 100},
        {.ndim = 4, .L = 4, .n_traj = 400},
        {.ndim = 4, .L = 6, .n_traj = 200},
        {.ndim = 4, .L = 8, .n_traj = 100},
    };

    constexpr int k_n_md   = 10;
    constexpr double k_tau = 1.0;
    constexpr int k_warmup = 30;

    std::printf(
        "Integrator: Omelyan2 (2 force evals/MD step)   tau=%.2f   n_md=%d\n\n", k_tau, k_n_md);
    std::printf("%-13s %-12s %-12s   %-12s %-14s %-12s %-10s\n",
                "ndim x L",
                "action",
                "real DOFs",
                "traj [s]",
                "DOF upd/s",
                "ratio (PQ/ref)",
                "accept");
    std::printf("%-13s %-12s %-12s   %-12s %-14s %-12s %-10s\n",
                "--------",
                "------",
                "---------",
                "--------",
                "---------",
                "--------------",
                "------");

    using Phi4    = act::Phi4<double>;
    using BoseGas = act::BoseGas<double>;

    Phi4 const phi4{.kappa = 0.18, .lambda = 1.145};
    BoseGas const bg{.mass = 1.0, .lambda = 1.0, .mu = 1.0};

    auto run_phi4 = [&](Case const& c) -> std::pair<double, double> {
        std::size_t const nd = static_cast<std::size_t>(c.ndim);
        std::size_t const L_ = static_cast<std::size_t>(c.L);
        Lattice<double>::SizeVec shape(nd, L_);
        Lattice<double> phi{shape};
        FastRng rng{42};
        alg::Hmc<Phi4, FastRng, Integ> hmc{phi4, phi, rng, {.tau = k_tau, .n_md = k_n_md}};
        for (int i = 0; i < k_warmup; ++i) {
            (void)hmc.step();
        }
        int acc       = 0;
        auto const t0 = bench_clock::now();
        for (int i = 0; i < c.n_traj; ++i) {
            acc += hmc.step().accepted ? 1 : 0;
        }
        double const per = seconds(bench_clock::now() - t0) / c.n_traj;
        return {per, static_cast<double>(acc) / c.n_traj};
    };

    auto run_bg = [&](Case const& c) -> std::pair<double, double> {
        std::size_t const nd = static_cast<std::size_t>(c.ndim);
        std::size_t const L_ = static_cast<std::size_t>(c.L);
        Lattice<std::complex<double>>::SizeVec shape(nd, L_);
        Lattice<std::complex<double>> phi{shape};
        FastRng rng{42};
        alg::Hmc<BoseGas, FastRng, Integ> hmc{bg, phi, rng, {.tau = k_tau, .n_md = k_n_md}};
        for (int i = 0; i < k_warmup; ++i) {
            (void)hmc.step();
        }
        int acc       = 0;
        auto const t0 = bench_clock::now();
        for (int i = 0; i < c.n_traj; ++i) {
            acc += hmc.step().accepted ? 1 : 0;
        }
        double const per = seconds(bench_clock::now() - t0) / c.n_traj;
        return {per, static_cast<double>(acc) / c.n_traj};
    };

    for (auto const& c : cases) {
        std::size_t const nd = static_cast<std::size_t>(c.ndim);
        std::size_t const L_ = static_cast<std::size_t>(c.L);
        std::size_t nsites   = 1;
        for (std::size_t k = 0; k < nd; ++k) {
            nsites *= L_;
        }
        std::size_t const dofs_phi4 = nsites;
        std::size_t const dofs_bg   = 2 * nsites;

        auto const [phi4_per, phi4_acc] = run_phi4(c);
        auto const [bg_per, bg_acc]     = run_bg(c);

        double const phi4_dof_per_s = static_cast<double>(dofs_phi4 * k_n_md) / phi4_per;
        double const bg_dof_per_s   = static_cast<double>(dofs_bg * k_n_md) / bg_per;
        double const ratio          = phi4_dof_per_s / bg_dof_per_s;

        std::printf("%dD L=%-3d   Phi4 (real)  %-12zu   %-12.3e %-13.1f M %-12s   %-10.3f\n",
                    c.ndim,
                    c.L,
                    dofs_phi4,
                    phi4_per,
                    phi4_dof_per_s / 1e6,
                    "—",
                    phi4_acc);
        std::printf("%dD L=%-3d   BoseGas (PQ) %-12zu   %-12.3e %-13.1f M %-12.2f   %-10.3f\n",
                    c.ndim,
                    c.L,
                    dofs_bg,
                    bg_per,
                    bg_dof_per_s / 1e6,
                    ratio,
                    bg_acc);
        std::printf("\n");
    }
}
