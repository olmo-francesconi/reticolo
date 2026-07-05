// bench_actions — kernel throughput per action on a hot random field.
//
// Calls `s_full`, `compute_force`, and (where available)
// `compute_force_and_kick` in tight loops with no HMC integrator, no
// Metropolis, no I/O. Reports per-call wall time and dofs/s (sites for
// scalar, links for link/matrix-link).

#include <reticolo/reticolo.hpp>

#include "_bench/hot_init.hpp"
#include "_bench/timing.hpp"

#include <array>
#include <complex>
#include <cstddef>
#include <cstdio>

namespace {

using reticolo::bench::consume;
using reticolo::bench::hot_init;
using reticolo::bench::time_per_call;

void print_header() {
    std::printf("%-12s %-16s %-10s %-22s %-12s %-14s\n",
                "ndim x L",
                "action",
                "dofs",
                "kernel",
                "wall [s]",
                "dof upd/s");
    std::printf("%-12s %-16s %-10s %-22s %-12s %-14s\n",
                "--------",
                "------",
                "----",
                "------",
                "--------",
                "---------");
}

void print_row(
    int ndim, int L, char const* action_name, std::size_t dofs, char const* kernel, double wall_s) {
    double const dof_per_s = static_cast<double>(dofs) / wall_s;
    std::printf("%dD L=%-3d   %-16s %-10zu %-22s %-12.3e %-12.2f M\n",
                ndim,
                L,
                action_name,
                dofs,
                kernel,
                wall_s,
                dof_per_s / 1e6);
}

template <class Action, class Field>
void bench_scalar_action(char const* name, int ndim, int L, Action const& action, Field& phi) {
    Field force{phi.indexing()};
    std::size_t const dofs = phi.nsites();

    double const t_sfull = time_per_call([&] { consume(action.s_full(phi)); });
    print_row(ndim, L, name, dofs, "s_full", t_sfull);

    double const t_force = time_per_call([&] { action.compute_force(phi, force); });
    print_row(ndim, L, name, dofs, "compute_force", t_force);

    if constexpr (requires {
                      action.compute_force_and_kick(phi, force, typename Action::value_type{0.123});
                  }) {
        double const t_fk = time_per_call(
            [&] { action.compute_force_and_kick(phi, force, typename Action::value_type{0.123}); });
        print_row(ndim, L, name, dofs, "compute_force_and_kick", t_fk);
    }
}

template <class Group>
void bench_wilson(char const* name,
                  int ndim,
                  int L,
                  double beta,
                  reticolo::MatrixLinkLattice<Group, double>& phi) {
    using Action = reticolo::action::Wilson<Group, double>;
    Action const action{.beta = beta};
    reticolo::MatrixLinkLattice<Group, double> force{phi.indexing()};
    std::size_t const dofs = phi.ndims() * phi.nsites();

    double const t_sfull = time_per_call([&] { consume(action.s_full(phi)); });
    print_row(ndim, L, name, dofs, "s_full", t_sfull);

    double const t_force = time_per_call([&] { action.compute_force(phi, force); });
    print_row(ndim, L, name, dofs, "compute_force", t_force);
}

struct Case {
    int ndim;
    int L;
};

void run_all() {
    using namespace reticolo;

    constexpr std::array<Case, 8> cases = {{
        {.ndim = 2, .L = 32},
        {.ndim = 2, .L = 64},
        {.ndim = 3, .L = 12},
        {.ndim = 3, .L = 16},
        {.ndim = 3, .L = 24},
        {.ndim = 4, .L = 4},
        {.ndim = 4, .L = 6},
        {.ndim = 4, .L = 8},
    }};

    print_header();

    for (auto const& c : cases) {
        std::size_t const nd = static_cast<std::size_t>(c.ndim);
        std::size_t const L_ = static_cast<std::size_t>(c.L);
        Lattice<double>::SizeVec const shape_s(nd, L_);

        FastRng rng{42};

        // Phi4
        {
            Lattice<double> phi{shape_s};
            hot_init(phi, rng);
            act::Phi4<double> const action{.kappa = 0.18, .lambda = 1.0};
            bench_scalar_action("Phi4", c.ndim, c.L, action, phi);
        }
        // Phi6
        {
            Lattice<double> phi{shape_s};
            hot_init(phi, rng);
            act::Phi6<double> const action{.kappa = 0.18, .lambda = 1.0, .g6 = 0.5};
            bench_scalar_action("Phi6", c.ndim, c.L, action, phi);
        }
        // SineGordon
        {
            Lattice<double> phi{shape_s};
            hot_init(phi, rng);
            act::SineGordon<double> const action{.kappa = 0.18, .alpha = 1.0};
            bench_scalar_action("SineGordon", c.ndim, c.L, action, phi);
        }
        // Xy
        {
            Lattice<double> phi{shape_s};
            hot_init(phi, rng);
            act::Xy<double> const action{.beta = 1.0};
            bench_scalar_action("Xy", c.ndim, c.L, action, phi);
        }
        // BoseGas (complex)
        if (c.ndim >= 3) {
            Lattice<std::complex<double>> phi{shape_s};
            hot_init(phi, rng);
            act::BoseGas<double> const action{.mass = 1.0, .lambda = 1.0, .mu = 0.9};
            std::size_t const dofs = phi.nsites();
            Lattice<std::complex<double>> force{phi.indexing()};
            double const t_sfull = time_per_call([&] { consume(action.s_full(phi)); });
            print_row(c.ndim, c.L, "BoseGas", dofs, "s_full", t_sfull);
            double const t_force = time_per_call([&] { action.compute_force(phi, force); });
            print_row(c.ndim, c.L, "BoseGas", dofs, "compute_force", t_force);
            double const t_simag = time_per_call([&] { consume(action.s_imag(phi)); });
            print_row(c.ndim, c.L, "BoseGas", dofs, "s_imag", t_simag);
            double const t_fimag = time_per_call([&] { action.compute_force_imag(phi, force); });
            print_row(c.ndim, c.L, "BoseGas", dofs, "compute_force_imag", t_fimag);
        }
        // Wilson<U1>
        {
            using F = MatrixLinkLattice<gauge_group::U1, double>;
            F::SizeVec const shape_m(nd, L_);
            F theta{shape_m};
            hot_init(theta, rng);
            bench_wilson<gauge_group::U1>("Wilson<U1>", c.ndim, c.L, 1.0, theta);
        }
        // Wilson<SU2>
        {
            using F = MatrixLinkLattice<gauge_group::SU2, double>;
            F::SizeVec const shape_m(nd, L_);
            F theta{shape_m};
            hot_init(theta, rng);
            bench_wilson<gauge_group::SU2>("Wilson<SU2>", c.ndim, c.L, 2.4, theta);
        }
        // Wilson<SU3>
        {
            using F = MatrixLinkLattice<gauge_group::SU3, double>;
            F::SizeVec const shape_m(nd, L_);
            F theta{shape_m};
            hot_init(theta, rng);
            bench_wilson<gauge_group::SU3>("Wilson<SU3>", c.ndim, c.L, 6.0, theta);
        }

        std::printf("\n");
    }
}

}  // namespace

int main() {
    reticolo::log::off();
    std::printf("ACTIONS — kernel throughput on hot random fields\n"
                "(kernels called in isolation, no MD loop)\n\n");
    run_all();
}
