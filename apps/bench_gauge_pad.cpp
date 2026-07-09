// Gauge cache-padding sweep: SU(3) Wilson force throughput across
//   volume L⁴ × threads × slabs-per-thread × {padded, packed}.
// Every (L, team, q) is chosen so the canonical partition divides the lattice
// with n_items % team == 0 (100% thread occupancy); the actual n_items and
// slabs/thread are reported so any snap is visible. Emits a JSON array on
// stdout. Force-only (isolates the staple kernel from the trajectory).
//
// Build: registered in apps/CMakeLists.txt (linux-gcc preset for OpenMP).
// Run:   ./bench_gauge_pad   (no args)

#include <reticolo/reticolo.hpp>

#include <chrono>
#include <cstdio>
#include <vector>

using reticolo::FastRng;
using reticolo::MatrixLinkLattice;
using reticolo::action::Wilson;
using SU3   = reticolo::math::group::SU3;
using Clock = std::chrono::steady_clock;

namespace {

constexpr std::size_t kLs[]   = {16, 24, 32, 48};
constexpr int kThreads[]      = {1, 2, 4, 8, 16};
constexpr int kQ[]            = {1, 2, 4};
constexpr double kBudget      = 0.5;  // timed seconds per config
constexpr int kWarm          = 3;

double time_force(Wilson<SU3, double> const& action,
                  MatrixLinkLattice<SU3, double> const& u,
                  MatrixLinkLattice<SU3, double>& f,
                  std::size_t d,
                  std::size_t ns) {
    for (int w = 0; w < kWarm; ++w) {
        action.force_into(u, f);
    }
    std::size_t calls = 0;
    auto const t0     = Clock::now();
    while (true) {
        action.force_into(u, f);
        ++calls;
        if ((calls & 3U) == 0) {
            double const el = std::chrono::duration<double>(Clock::now() - t0).count();
            if (el >= kBudget) {
                return static_cast<double>(d * ns) * static_cast<double>(calls) / el / 1e6;
            }
        }
    }
}

}  // namespace

int main() {
    reticolo::log::off();
    Wilson<SU3, double> const action{.beta = 6.0};

    std::printf("[\n");
    bool first = true;
    for (std::size_t L : kLs) {
        for (int pad = 0; pad <= 1; ++pad) {
            reticolo::g_gauge_link_padding = (pad != 0);
            MatrixLinkLattice<SU3, double> u{{L, L, L, L}};
            MatrixLinkLattice<SU3, double> f{{L, L, L, L}};
            reticolo::g_gauge_link_padding = false;

            std::size_t const ns   = u.nsites();
            std::size_t const span = u.link_span();
            std::size_t const d    = u.ndims();
            FastRng rng{2024};
            for (std::size_t mu = 0; mu < d; ++mu) {
                SU3::sample_algebra_slab(u.mu_block_data(mu), rng, span, ns);
            }

            for (int team : kThreads) {
                for (int q : kQ) {
                    reticolo::exec::team_scope const ts{team};
                    reticolo::exec::slab_scope const ss{q};
                    auto const part = reticolo::exec::partition(u);
                    double const mlps = time_force(action, u, f, d, ns);
                    if (!first) {
                        std::printf(",\n");
                    }
                    first = false;
                    std::printf(
                        "  {\"L\":%zu,\"nsites\":%zu,\"threads\":%d,\"q\":%d,\"pad\":%d,"
                        "\"n_items\":%zu,\"slabs_per_thread\":%.3f,\"mlps\":%.3f}",
                        L, ns, team, q, pad, part.n_items,
                        static_cast<double>(part.n_items) / team, mlps);
                    std::fflush(stdout);
                }
            }
        }
    }
    std::printf("\n]\n");
    return 0;
}
