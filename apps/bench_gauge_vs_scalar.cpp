// Head-to-head HMC throughput: scalar Phi4 (polynomial), scalar SineGordon
// (one transcendental per site), and compact-U(1) Wilson (one transcendental
// per plaquette → (d-1)/2 per link). Same integrator (Omelyan2), same tau
// and n_md, matched lattice shapes. Reports trajectory wall time, dof
// updates per second, and an estimated MFLOPS so two actions with different
// per-dof arithmetic densities can be compared on equal footing.
//
// Flop accounting per dof per force evaluation (drift + kick fold in trivially):
//
//   Phi4         neighbour sum (2d) + polynomial (~7)             ≈ 2d + 7
//   SineGordon   neighbour sum (2d) + 1 sin + ~3 polynomial      ≈ 2d + 3 + k_sin
//   CompactU1    plaquette-centric scatter; each plaquette costs
//                3 adds (plaq angle) + 1 sin + 1 mul + 4 link
//                scatters (4 mul-add) and touches 4 links →
//                amortised per link: (d-1)·(3 + k_sin + 9)/2
//
//   where k_sin = approximate libm sin/cos cost (~15 ops on modern hardware).
//
// Omelyan2 per trajectory: (2*n_md + 1) force evals + 2*n_md drift updates
// (2 ops/dof each). The constant front matter (momentum sampling, H()
// computation) is small enough at production n_md to fold into the noise.

#include <reticolo/reticolo.hpp>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace {

using bench_clock = std::chrono::steady_clock;

double seconds(bench_clock::duration d) {
    return std::chrono::duration<double>(d).count();
}

// Approximate cost of a single libm sin/cos call on modern hardware in
// "effective" flops. ~12-20 cycle latency; we pick 15 as a defensible mid-
// point that lets MFLOPS values stay comparable across transcendental and
// pure-arithmetic actions.
constexpr double k_sin_flops = 15.0;

double phi4_flops_per_force(int ndim) {
    return (2.0 * ndim) + 7.0;
}

double sg_flops_per_force(int ndim) {
    return (2.0 * ndim) + 3.0 + k_sin_flops;
}

double u1_flops_per_force_per_link(int ndim) {
    double const flops_per_plaq = 3.0 + k_sin_flops + 1.0 + 8.0;
    double const plaqs_per_link = static_cast<double>(ndim - 1);
    return 0.5 * plaqs_per_link * flops_per_plaq;
}

double traj_flops_per_dof(double flops_per_force, int n_md) {
    return ((2.0 * n_md) + 1.0) * flops_per_force + (2.0 * n_md * 2.0);
}

struct Case {
    int ndim;
    int L;
    int n_traj;
};

struct Row {
    char const* label;
    std::size_t dofs;
    double flops_per_force;
    double traj_seconds;
    double accept;
};

void print_header() {
    std::printf("%-14s %-10s %-10s   %-12s %-14s %-12s %-10s\n",
                "ndim x L · action",
                "dofs",
                "f/force",
                "traj [s]",
                "dof upd/s",
                "MFLOPS",
                "accept");
    std::printf("%-14s %-10s %-10s   %-12s %-14s %-12s %-10s\n",
                "-----------------",
                "----",
                "-------",
                "--------",
                "---------",
                "------",
                "------");
}

void print_row(Row const& r, int ndim, int L, int n_md) {
    double const dof_per_s = static_cast<double>(r.dofs) * n_md / r.traj_seconds;
    double const flops_per_s =
        static_cast<double>(r.dofs) * traj_flops_per_dof(r.flops_per_force, n_md) / r.traj_seconds;
    std::printf("%dD L=%-3d %-10s %-10zu %-10.1f   %-12.3e %-13.1f M %-12.1f %-10.3f\n",
                ndim,
                L,
                r.label,
                r.dofs,
                r.flops_per_force,
                r.traj_seconds,
                dof_per_s / 1e6,
                flops_per_s / 1e6,
                r.accept);
}

}  // namespace

int main() {
    using namespace reticolo;
    log::off();
    using Integ      = alg::integ::Omelyan2;
    using GaugeInteg = alg::integ::Omelyan2;

    std::vector<Case> const cases = {
        {.ndim = 3, .L = 8, .n_traj = 200},
        {.ndim = 3, .L = 12, .n_traj = 200},
        {.ndim = 3, .L = 16, .n_traj = 100},
        {.ndim = 4, .L = 4, .n_traj = 400},
        {.ndim = 4, .L = 6, .n_traj = 200},
        {.ndim = 4, .L = 8, .n_traj = 100},
    };

    constexpr int k_n_md      = 10;
    constexpr double k_tau    = 1.0;
    constexpr int k_warmup    = 30;
    constexpr double k_kappa  = 0.18;
    constexpr double k_lambda = 1.145;
    constexpr double k_alpha  = 1.0;
    constexpr double k_beta   = 1.01;

    std::printf(
        "Integrator: Omelyan2 (2 force evals/MD step)   tau=%.2f   n_md=%d\n", k_tau, k_n_md);
    std::printf("Flop accounting: k_sin ≈ %.1f ops per sin/cos call\n\n", k_sin_flops);

    using ScalarPhi4 = act::Phi4<double>;
    using ScalarSG   = act::SineGordon<double>;
    using GaugeU1    = action::CompactU1<double>;

    ScalarPhi4 const phi4{.kappa = k_kappa, .lambda = k_lambda};
    ScalarSG const sg{.kappa = k_kappa, .alpha = k_alpha};
    GaugeU1 const u1{.beta = k_beta};

    print_header();

    for (auto const& c : cases) {
        std::size_t const nd = static_cast<std::size_t>(c.ndim);
        std::size_t const L_ = static_cast<std::size_t>(c.L);
        Lattice<double>::SizeVec shape_s(nd, L_);
        LinkLattice<double>::SizeVec shape_g(nd, L_);

        // Phi4.
        {
            Lattice<double> phi{shape_s};
            FastRng rng{42};
            alg::Hmc<ScalarPhi4, FastRng, Integ> hmc{
                phi4, phi, rng, {.tau = k_tau, .n_md = k_n_md}};
            for (int i = 0; i < k_warmup; ++i) {
                (void)hmc.step();
            }
            int acc       = 0;
            auto const t0 = bench_clock::now();
            for (int i = 0; i < c.n_traj; ++i) {
                acc += hmc.step().accepted ? 1 : 0;
            }
            print_row({.label           = "Phi4",
                       .dofs            = phi.nsites(),
                       .flops_per_force = phi4_flops_per_force(c.ndim),
                       .traj_seconds    = seconds(bench_clock::now() - t0) / c.n_traj,
                       .accept          = static_cast<double>(acc) / c.n_traj},
                      c.ndim,
                      c.L,
                      k_n_md);
        }

        // SineGordon.
        {
            Lattice<double> phi{shape_s};
            FastRng rng{42};
            alg::Hmc<ScalarSG, FastRng, Integ> hmc{sg, phi, rng, {.tau = k_tau, .n_md = k_n_md}};
            for (int i = 0; i < k_warmup; ++i) {
                (void)hmc.step();
            }
            int acc       = 0;
            auto const t0 = bench_clock::now();
            for (int i = 0; i < c.n_traj; ++i) {
                acc += hmc.step().accepted ? 1 : 0;
            }
            print_row({.label           = "SineGordon",
                       .dofs            = phi.nsites(),
                       .flops_per_force = sg_flops_per_force(c.ndim),
                       .traj_seconds    = seconds(bench_clock::now() - t0) / c.n_traj,
                       .accept          = static_cast<double>(acc) / c.n_traj},
                      c.ndim,
                      c.L,
                      k_n_md);
        }

        // CompactU1 (gauge).
        {
            LinkLattice<double> theta{shape_g, 0.0};
            FastRng rng{42};
            alg::Hmc<GaugeU1, FastRng, GaugeInteg, LinkLattice<double>> hmc{
                u1, theta, rng, {.tau = k_tau, .n_md = k_n_md}};
            for (int i = 0; i < k_warmup; ++i) {
                (void)hmc.step();
            }
            int acc       = 0;
            auto const t0 = bench_clock::now();
            for (int i = 0; i < c.n_traj; ++i) {
                acc += hmc.step().accepted ? 1 : 0;
            }
            print_row({.label           = "CompactU1",
                       .dofs            = theta.nlinks(),
                       .flops_per_force = u1_flops_per_force_per_link(c.ndim),
                       .traj_seconds    = seconds(bench_clock::now() - t0) / c.n_traj,
                       .accept          = static_cast<double>(acc) / c.n_traj},
                      c.ndim,
                      c.L,
                      k_n_md);
        }
        std::printf("\n");
    }
}
