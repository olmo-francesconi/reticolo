// bench_integrators — full MD-trajectory throughput per (action, integrator)
// at fixed (τ = 1.0, n_md = 20). No Metropolis, no momentum resampling
// between trajectories — back-to-back calls to `Integrator::run` against a
// freshly-sampled momentum + hot field. Reveals the per-integrator wall-
// time ratio for each action — for matrix-link Wilson<SU{2,3}> the drift
// (group exp) is comparable to one force eval, so the L : O2 : O4 ratio
// deviates from the naive 1 : 2 : 4 that scalar actions show.

#include "_bench/hot_init.hpp"
#include "_bench/timing.hpp"

#include <reticolo/reticolo.hpp>

#include <array>
#include <complex>
#include <cstddef>
#include <cstdio>
#include <type_traits>

namespace {

using reticolo::bench::hot_init;
using reticolo::bench::time_per_call;

constexpr double k_tau = 1.0;
constexpr int k_n_md   = 20;

constexpr int k_leapfrog_force_evals_per_traj = k_n_md + 1;
constexpr int k_omelyan2_force_evals_per_traj = (2 * k_n_md) + 1;
constexpr int k_omelyan4_force_evals_per_traj = (4 * k_n_md) + 1;

void print_header() {
    std::printf("%-12s %-16s %-12s %-10s %-12s %-14s\n",
                "ndim x L",
                "action",
                "integrator",
                "dofs",
                "wall/traj",
                "force evals/s");
    std::printf("%-12s %-16s %-12s %-10s %-12s %-14s\n",
                "--------",
                "------",
                "----------",
                "----",
                "---------",
                "-------------");
}

void print_row(int ndim,
               int L,
               char const* action,
               char const* integ,
               std::size_t dofs,
               double wall_s,
               int force_evals_per_traj) {
    double const force_evals_per_s =
        static_cast<double>(force_evals_per_traj) / wall_s;
    std::printf("%dD L=%-3d   %-16s %-12s %-10zu %-12.3e %-14.1f\n",
                ndim,
                L,
                action,
                integ,
                dofs,
                wall_s,
                force_evals_per_s);
}

template <class Action, class Field>
void bench_one(int ndim,
               int L,
               char const* action_name,
               Action const& action,
               Field& phi,
               std::size_t dofs) {
    using namespace reticolo;
    Field mom{phi.indexing()};
    Field force{phi.indexing()};

    // Sample a momentum once outside the timed loop. Each trajectory
    // call reads from `mom` and writes back; back-to-back calls preserve
    // detailed-balance-irrelevant state.
    FastRng rng{777};
    using F = typename Field::value_type;
    if constexpr (requires { typename Field::group_type; }) {
        using G              = typename Field::group_type;
        std::size_t const ns = mom.nsites();
        for (std::size_t mu = 0; mu < mom.ndims(); ++mu) {
            G::sample_algebra_slab(mom.mu_block_data(mu), rng, ns);
        }
    } else if constexpr (std::is_same_v<F, double>) {
        rng.normal_fill(mom.data(), flat_size(mom));
    } else if constexpr (std::is_same_v<F, std::complex<double>>) {
        rng.normal_fill(reinterpret_cast<double*>(mom.data()),
                        2 * flat_size(mom));
    }

    using namespace reticolo::alg::integ;
    {
        double const t = time_per_call([&] {
            Leapfrog::run(action, phi, mom, force, k_tau, k_n_md);
        });
        print_row(ndim, L, action_name, "Leapfrog", dofs, t,
                  k_leapfrog_force_evals_per_traj);
    }
    {
        double const t = time_per_call([&] {
            Omelyan2::run(action, phi, mom, force, k_tau, k_n_md);
        });
        print_row(ndim, L, action_name, "Omelyan2", dofs, t,
                  k_omelyan2_force_evals_per_traj);
    }
    {
        double const t = time_per_call([&] {
            Omelyan4::run(action, phi, mom, force, k_tau, k_n_md);
        });
        print_row(ndim, L, action_name, "Omelyan4", dofs, t,
                  k_omelyan4_force_evals_per_traj);
    }
}

struct Case {
    int ndim;
    int L;
};

void run_all() {
    using namespace reticolo;

    constexpr std::array<Case, 5> cases = {{
        {.ndim = 3, .L = 12},
        {.ndim = 3, .L = 16},
        {.ndim = 4, .L = 4},
        {.ndim = 4, .L = 6},
        {.ndim = 4, .L = 8},
    }};

    print_header();

    for (auto const& c : cases) {
        std::size_t const nd = static_cast<std::size_t>(c.ndim);
        std::size_t const L_ = static_cast<std::size_t>(c.L);
        Lattice<double>::SizeVec const shape_s(nd, L_);
        LinkLattice<double>::SizeVec const shape_l(nd, L_);

        FastRng init_rng{42};

        // Phi4
        {
            Lattice<double> phi{shape_s};
            hot_init(phi, init_rng);
            act::Phi4<double> const action{.kappa = 0.18, .lambda = 1.0};
            bench_one(c.ndim, c.L, "Phi4", action, phi, phi.nsites());
        }
        // Phi6
        {
            Lattice<double> phi{shape_s};
            hot_init(phi, init_rng);
            act::Phi6<double> const action{.kappa = 0.18, .lambda = 1.0, .g6 = 0.5};
            bench_one(c.ndim, c.L, "Phi6", action, phi, phi.nsites());
        }
        // SineGordon
        {
            Lattice<double> phi{shape_s};
            hot_init(phi, init_rng);
            act::SineGordon<double> const action{.kappa = 0.18, .alpha = 1.0};
            bench_one(c.ndim, c.L, "SineGordon", action, phi, phi.nsites());
        }
        // BoseGas (complex)
        {
            Lattice<std::complex<double>> phi{shape_s};
            hot_init(phi, init_rng);
            act::BoseGas<double> const action{
                .mass = 1.0, .lambda = 1.0, .mu = 0.9};
            bench_one(c.ndim, c.L, "BoseGas", action, phi, phi.nsites());
        }
        // CompactU1 (LinkLattice<double>)
        {
            LinkLattice<double> theta{shape_l, 0.0};
            hot_init(theta, init_rng);
            action::CompactU1<double> const action{.beta = 1.0};
            bench_one(c.ndim, c.L, "CompactU1", action, theta, theta.nlinks());
        }
        // Wilson<SU2>
        {
            using F = MatrixLinkLattice<gauge_group::SU2, double>;
            F::SizeVec const shape_m(nd, L_);
            F theta{shape_m};
            hot_init(theta, init_rng);
            action::Wilson<gauge_group::SU2, double> const action{.beta = 2.4};
            bench_one(c.ndim, c.L, "Wilson<SU2>", action, theta,
                      theta.ndims() * theta.nsites());
        }
        // Wilson<SU3>
        {
            using F = MatrixLinkLattice<gauge_group::SU3, double>;
            F::SizeVec const shape_m(nd, L_);
            F theta{shape_m};
            hot_init(theta, init_rng);
            action::Wilson<gauge_group::SU3, double> const action{.beta = 6.0};
            bench_one(c.ndim, c.L, "Wilson<SU3>", action, theta,
                      theta.ndims() * theta.nsites());
        }
        std::printf("\n");
    }
}

}  // namespace

int main() {
    std::printf(
        "INTEGRATORS — full MD-trajectory throughput at tau=%.2f, n_md=%d\n"
        "(no Metropolis, no momentum resampling between trajectories)\n\n",
        k_tau,
        k_n_md);
    run_all();
}
