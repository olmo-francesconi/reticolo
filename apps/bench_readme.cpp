// bench_readme — the headline numbers quoted in README's Performance
// section: RNG generation throughput (all three RNGs) and force-kernel
// throughput for a scalar baseline (Phi4) vs the SU(2)/SU(3) gauge actions.
//
// Deliberately narrow: bench_rng / bench_actions sweep every field type and
// action; this one prints only what the README tables show, so the quoted
// numbers are reproducible with a single binary. Single-threaded kernels;
// for each cell we report the mean and 5th/95th-percentile throughput over
// many batch samples (see _bench/timing.hpp time_distribution).

#include <reticolo/reticolo.hpp>

#include "_bench/hot_init.hpp"
#include "_bench/timing.hpp"

#include <array>
#include <cstddef>
#include <cstdio>
#include <vector>

namespace {

using reticolo::bench::consume;
using reticolo::bench::hot_init;
using reticolo::bench::Stats;
using reticolo::bench::time_distribution;

constexpr std::array<int, 7> k_sizes = {4, 6, 8, 12, 16, 20, 24};

// Throughput [M ops/s] from per-call time stats: mean from the mean time,
// and the percentile band inverted (slowest time -> 5th-pct throughput).
void print_tp(char const* a, char const* b, std::size_t ops, Stats s) {
    double const n = static_cast<double>(ops);
    std::printf("%-14s %-9s %12zu %10.1f %10.1f %10.1f\n",
                a,
                b,
                ops,
                (n / s.mean) / 1e6,
                (n / s.p95) / 1e6,
                (n / s.p05) / 1e6);
}

// ---- RNG generation throughput ----------------------------------------------
template <class Rng>
void bench_rng(char const* name) {
    constexpr std::size_t n = 1UL << 20;  // ~1M draws per call
    std::vector<double> buf(n);
    Rng rng{2024};

    Stats const uni = time_distribution([&] {
        for (std::size_t i = 0; i < n; ++i) {
            buf[i] = rng.uniform();
        }
        consume(buf[n - 1]);
    });
    print_tp(name, "uniform", n, uni);

    Stats const nrm = time_distribution([&] { rng.normal_fill(buf.data(), n); });
    print_tp(name, "gaussian", n, nrm);
}

// ---- Force-kernel throughput -------------------------------------------------
void bench_phi4(int L) {
    using namespace reticolo;
    Lattice<double>::SizeVec const shape(4, static_cast<std::size_t>(L));
    Lattice<double> phi{shape};
    FastRng rng{42};
    hot_init(phi, rng);
    act::Phi4<double> const action{.kappa = 0.18, .lambda = 1.0};
    Lattice<double> force{phi.indexing()};

    char lat[8];
    std::snprintf(lat, sizeof(lat), "%d^4", L);
    Stats const s = time_distribution([&] { action.compute_force(phi, force); });
    print_tp("Phi4", lat, phi.nsites(), s);
}

template <class Group>
void bench_wilson(char const* name, int L, double beta) {
    using namespace reticolo;
    using Field = MatrixLinkLattice<Group, double>;
    typename Field::SizeVec const shape(4, static_cast<std::size_t>(L));
    Field u{shape};
    FastRng rng{42};
    hot_init(u, rng);
    act::Wilson<Group, double> const action{.beta = beta};
    Field force{u.indexing()};

    char lat[8];
    std::snprintf(lat, sizeof(lat), "%d^4", L);
    Stats const s = time_distribution([&] { action.compute_force(u, force); });
    print_tp(name, lat, u.ndims() * u.nsites(), s);
}

}  // namespace

int main() {
    using namespace reticolo;
    log::off();

    std::printf("RNG — generation throughput [M draws/s]\n\n");
    std::printf("%-14s %-9s %12s %10s %10s %10s\n", "rng", "draw", "draws", "mean", "p05", "p95");
    std::printf("%-14s %-9s %12s %10s %10s %10s\n", "---", "----", "-----", "----", "---", "---");
    bench_rng<FastRng>("FastRng");
    bench_rng<PhiloxRng>("Philox4x32");
    bench_rng<RanlxdRng>("Ranlxd");
    bench_rng<Mt19937Rng>("Mt19937_64");

    std::printf("\n\ncompute_force — throughput [M dof-updates/s]\n\n");
    std::printf(
        "%-14s %-9s %12s %10s %10s %10s\n", "action", "lattice", "dofs", "mean", "p05", "p95");
    std::printf(
        "%-14s %-9s %12s %10s %10s %10s\n", "------", "-------", "----", "----", "---", "---");
    for (int L : k_sizes) {
        bench_phi4(L);
    }
    for (int L : k_sizes) {
        bench_wilson<math::group::SU2>("Wilson<SU2>", L, 2.4);
    }
    for (int L : k_sizes) {
        bench_wilson<math::group::SU3>("Wilson<SU3>", L, 6.0);
    }
}
