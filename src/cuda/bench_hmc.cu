// First P100 perf baseline for Phi4 HMC (f64) on the device: per-trajectory
// wall time and trajectories/second for cuda::Hmc across a range of volumes.
// Compared offline against the same Phi4 HMC config run on the Mac CPU core
// (different machine — a rough but useful first comparison).
//
// Same physics as the CPU bench: Phi4 (kappa=0.18, lambda=1.0), Leapfrog,
// tau=1.0, n_md=10. The first step captures the MD graph (excluded from timing
// by the warm-up).

#include <reticolo/action/phi4.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/cuda/actions/phi4.hpp>
#include <reticolo/cuda/device_action.cuh>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/hmc.cuh>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace {

using reticolo::Lattice;
using reticolo::action::Phi4;
using reticolo::cuda::DeviceAction;
using reticolo::cuda::DeviceField;

using clock_type = std::chrono::steady_clock;

constexpr double kKappa  = 0.18;
constexpr double kLambda = 1.0;
constexpr double kTau    = 1.0;
constexpr int kNmd       = 10;
constexpr int kWarmup    = 3;
constexpr int kIters     = 20;

double bench_gpu(std::vector<std::size_t> const& shape) {
    DeviceField<double> field{shape};
    Lattice<double> const zero{shape};
    field.copy_from_host(zero);
    cudaDeviceSynchronize();

    Phi4<double> action{.kappa = kKappa, .lambda = kLambda};
    DeviceAction<Phi4<double>, DeviceField<double>> dact{action, field.topology()};
    reticolo::cuda::Hmc<DeviceAction<Phi4<double>, DeviceField<double>>> hmc{
        std::move(dact), field, kTau, kNmd, 12345ULL};

    hmc.run(kWarmup);  // first replay captures the full-trajectory graph
    hmc.sync();
    auto const t0 = clock_type::now();
    hmc.run(kIters);  // host-free: kIters graph replays, no per-step sync
    hmc.sync();
    return std::chrono::duration<double>(clock_type::now() - t0).count() / kIters;
}

}  // namespace

int main() {
    reticolo::log::off();

    std::printf("P100 — Phi4 HMC (f64), tau=%.1f n_md=%d\n", kTau, kNmd);
    std::printf("%-12s %-12s   %-13s %-14s\n", "dims", "V", "GPU [ms/traj]", "traj/s");
    std::printf("------------------------------------------------------\n");

    struct Case {
        int ndim;
        int L;
    };
    Case const cases[] = {{4, 4}, {4, 8}, {4, 12}, {4, 16}, {4, 24}, {4, 32}};

    for (Case const c : cases) {
        std::vector<std::size_t> shape(static_cast<std::size_t>(c.ndim),
                                       static_cast<std::size_t>(c.L));
        std::size_t v = 1;
        for (std::size_t s : shape) {
            v *= s;
        }
        double const t = bench_gpu(shape);
        std::printf("%dD x L=%-6d %-12zu   %-13.4f %-14.1f\n", c.ndim, c.L, v, t * 1e3, 1.0 / t);
    }
    return 0;
}
