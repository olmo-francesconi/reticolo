// P100 perf survey across every action ported to the device so far: Phi4
// (f64 + f32), Phi6, SineGordon, XY (scalar / ScalarLayout) and CompactU1
// (gauge / LinkLayout). Each runs the host-free cuda::Hmc through the unified
// DeviceAction, so this also exercises both access patterns end-to-end. Reports
// per-trajectory wall time and trajectories/second. Leapfrog, tau=1.0, n_md=10.
//
// DOF = field.size(): nsites for the scalar actions, ndim·nsites for the gauge
// link field — the gauge row does ~ndim× the elementwise work at equal V, so
// compare within an action across V, and across actions at equal DOF.

#include <reticolo/action/gauge/wilson.hpp>
#include <reticolo/action/gauge/wilson.hpp>
#include <reticolo/action/site/phi4.hpp>
#include <reticolo/action/site/phi6.hpp>
#include <reticolo/action/site/sine_gordon.hpp>
#include <reticolo/action/bond/xy.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/cuda/actions/gauge/wilson.hpp>
#include <reticolo/cuda/actions/site/phi4.hpp>
#include <reticolo/cuda/actions/site/phi6.hpp>
#include <reticolo/cuda/actions/site/sine_gordon.hpp>
#include <reticolo/cuda/actions/gauge/wilson.hpp>
#include <reticolo/cuda/actions/bond/xy.hpp>
#include <reticolo/cuda/device_action.cuh>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/gauge/su2_device.cuh>
#include <reticolo/cuda/gauge/su3_device.cuh>
#include <reticolo/cuda/hmc.cuh>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace {

using reticolo::cuda::DeviceAction;
using reticolo::cuda::DeviceField;
using reticolo::cuda::LinkLayout;
using reticolo::cuda::MatrixLayout;
namespace act = reticolo::action;

using clock_type = std::chrono::steady_clock;

constexpr double kTau = 1.0;
constexpr int kNmd    = 10;
constexpr int kWarmup = 3;
constexpr int kIters  = 20;

// One host-free HMC run over a zero (cold) config — physics is irrelevant here,
// the kernels do the same work regardless of contents. Returns seconds/traj.
template <class HostAction, class Field>
void bench(char const* label, std::vector<std::size_t> const& shape, HostAction action) {
    using T = typename Field::value_type;
    Field field{shape};
    std::vector<T> const zero(field.size(), T{0});
    field.copy_from_host(zero.data());
    cudaDeviceSynchronize();

    DeviceAction<HostAction, Field> dact{action, field.topology()};
    reticolo::cuda::Hmc<DeviceAction<HostAction, Field>, reticolo::alg::integ::Leapfrog, Field> hmc{
        std::move(dact), field, kTau, kNmd};

    hmc.run(kWarmup);  // first replay captures the full-trajectory graph
    hmc.sync();
    auto const t0 = clock_type::now();
    hmc.run(kIters);
    hmc.sync();
    double const t = std::chrono::duration<double>(clock_type::now() - t0).count() / kIters;

    std::size_t v = 1;
    for (std::size_t s : shape) {
        v *= s;
    }
    std::printf("%-14s %dD L=%-3d %-9zu %-9zu %-11.4f %-10.1f\n",
                label,
                static_cast<int>(shape.size()),
                static_cast<int>(shape[0]),
                v,
                field.size(),
                t * 1e3,
                1.0 / t);
}

}  // namespace

int main() {
    reticolo::log::off();

    std::printf("P100 — HMC throughput (host-free), Leapfrog tau=%.1f n_md=%d\n", kTau, kNmd);
    std::printf(
        "%-14s %-9s %-9s %-9s %-11s %-10s\n", "action", "dims", "V", "DOF", "ms/traj", "traj/s");
    std::printf("---------------------------------------------------------------------\n");

    int const ls[] = {8, 16, 32};
    for (int l : ls) {
        std::vector<std::size_t> const s4(4, static_cast<std::size_t>(l));
        bench<act::Phi4<double>, DeviceField<double>>(
            "Phi4<f64>", s4, {.kappa = 0.18, .lambda = 1.0});
        bench<act::Phi4<float>, DeviceField<float>>(
            "Phi4<f32>", s4, {.kappa = 0.18F, .lambda = 1.0F});
        bench<act::Phi6<double>, DeviceField<double>>(
            "Phi6<f64>", s4, {.kappa = 0.18, .lambda = 1.0, .g6 = 0.1});
        bench<act::SineGordon<double>, DeviceField<double>>(
            "SineGordon<f64>", s4, {.kappa = 0.18, .alpha = 1.0});
        bench<act::Xy<double>, DeviceField<double>>("Xy<f64>", s4, {.beta = 0.5});
        bench<act::Wilson<reticolo::gauge_group::U1, double>, DeviceField<double, LinkLayout>>(
            "Wilson<U1><f64>", s4, {.beta = 1.0});
        bench<act::Wilson<reticolo::gauge_group::SU2, double>,
              DeviceField<double, MatrixLayout<reticolo::gauge_group::SU2>>>(
            "Wilson<SU2>", s4, {.beta = 2.4});
        bench<act::Wilson<reticolo::gauge_group::SU3, double>,
              DeviceField<double, MatrixLayout<reticolo::gauge_group::SU3>>>(
            "Wilson<SU3>", s4, {.beta = 6.0});
        std::printf("---------------------------------------------------------------------\n");
    }
    return 0;
}
