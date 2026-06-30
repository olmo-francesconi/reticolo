// Single-config profiling driver for the Nsight tools. Runs ONE (action, L) per
// invocation so `nsys` attributes kernel time per configuration (the all-in-one
// bench_cuda_hmc would aggregate across configs). Two modes:
//
//   default      host-free cuda::Hmc trajectories (graph replay) — the real
//                workload nsys traces; emits one JSON line with wall-clock
//                throughput (ms/traj, traj/s).
//   --force-only loop of EAGER DeviceAction::compute_force launches (no graph) —
//                a clean single-kernel target for `ncu` (DRAM %/occupancy/
//                roofline); emits a rough streaming-model effective GB/s.
//
// Links reticolo::cuda only (no umbrella/io) — kernel names alone give the nsys
// breakdown (stencil_kernel<Phi4ForceFunctor>, su_plaq_force_kernel<SU3Device>,
// reduce_sumsq_*, axpy_*, ...). Args: --action=phi4|su3 --size=L [--ndim=4]
// [--n_md=10] [--iters=30] [--force-only].

#include <reticolo/action/phi4.hpp>
#include <reticolo/action/wilson.hpp>
#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/cuda/actions/phi4.hpp>
#include <reticolo/cuda/actions/wilson.hpp>
#include <reticolo/cuda/device_action.cuh>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/gauge/su3_device.cuh>
#include <reticolo/cuda/gauge_sun.cuh>
#include <reticolo/cuda/hmc.cuh>

#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace {

using reticolo::cuda::DeviceAction;
using reticolo::cuda::DeviceField;
using reticolo::cuda::MatrixLayout;
namespace act    = reticolo::action;
using clock_type = std::chrono::steady_clock;

constexpr double kPeakGBps = 732.0;  // Tesla P100 HBM2 peak (≈732 GB/s)

template <class HostAction, class Field>
void run_config(char const* label,
                std::vector<std::size_t> const& shape,
                HostAction action,
                int n_md,
                int iters,
                bool force_only) {
    using T = typename Field::value_type;
    Field field{shape};
    std::vector<T> const zero(field.size(), T{0});
    field.copy_from_host(zero.data());
    cudaDeviceSynchronize();

    std::size_t v = 1;
    for (std::size_t s : shape) {
        v *= s;
    }
    auto const dof   = static_cast<double>(field.size());
    double const gib = (dof * sizeof(T)) / (1024.0 * 1024.0 * 1024.0);

    DeviceAction<HostAction, Field> dact{action, field.topology()};

    if (force_only) {
        // Eager force launches — clean ncu target, one kernel per call.
        Field force{field.topology()};
        dact.compute_force(field, force);  // warm
        cudaDeviceSynchronize();
        auto const t0 = clock_type::now();
        for (int i = 0; i < iters; ++i) {
            dact.compute_force(field, force);
        }
        cudaDeviceSynchronize();
        double const t = std::chrono::duration<double>(clock_type::now() - t0).count() / iters;
        // Streaming model: read field once + write force once = 2·dof·sizeof(T).
        double const gbps = (2.0 * dof * sizeof(T)) / t / 1e9;
        std::printf(
            "{\"action\":\"%s\",\"mode\":\"force\",\"ndim\":%zu,\"L\":%zu,\"V\":%zu,\"dof\":%.0f,"
            "\"field_GiB\":%.4f,\"us_per_force\":%.4f,\"eff_GBps\":%.1f,\"pct_peak\":%.1f}\n",
            label,
            shape.size(),
            shape[0],
            v,
            dof,
            gib,
            t * 1e6,
            gbps,
            100.0 * gbps / kPeakGBps);
        return;
    }

    // Per-atom breakdown via CUDA events (eager, isolated): nsys is absent and
    // ncu's counters are locked on the managed host, so this is the portable way
    // to see where the trajectory time goes. Each atom is the exact kernel the MD
    // loop calls; the Mac analysis weights them by per-trajectory call counts
    // (force ×n_md, axpy ×2·n_md, s_full ×2, sample ×1).
    Field mom{field.topology()};
    Field force{field.topology()};
    reticolo::cuda::DeviceBuffer<std::uint64_t> traj{1};
    std::uint64_t const traj0 = 0;
    traj.copy_from_host(&traj0);
    cudaDeviceSynchronize();
    auto const n = static_cast<long>(field.size());

    auto time_us = [](auto&& fn) {
        cudaEvent_t a = nullptr;
        cudaEvent_t b = nullptr;
        cudaEventCreate(&a);
        cudaEventCreate(&b);
        fn();  // warm
        cudaDeviceSynchronize();
        constexpr int reps = 20;
        cudaEventRecord(a);
        for (int i = 0; i < reps; ++i) {
            fn();
        }
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.0F;
        cudaEventElapsedTime(&ms, a, b);
        cudaEventDestroy(a);
        cudaEventDestroy(b);
        return (1e3 * static_cast<double>(ms)) / reps;  // µs/call
    };

    double const us_force = time_us([&] { dact.compute_force(field, force); });
    double const us_sfull = time_us([&] { (void)dact.s_full(field); });
    double const us_axpy  = time_us([&] { reticolo::cuda::kick_add(mom, force, 0.1); });
    double const us_sample =
        time_us([&] { dact.sample_momenta(mom.data(), n, 1234ULL, traj.data(), nullptr); });

    reticolo::cuda::Hmc<DeviceAction<HostAction, Field>, reticolo::alg::integ::Leapfrog, Field> hmc{
        std::move(dact), field, 1.0, n_md};
    hmc.run(3);  // capture the trajectory graph
    hmc.sync();
    auto const t0 = clock_type::now();
    hmc.run(iters);
    hmc.sync();
    double const t = std::chrono::duration<double>(clock_type::now() - t0).count() / iters;
    std::printf("{\"action\":\"%s\",\"mode\":\"hmc\",\"ndim\":%zu,\"L\":%zu,\"V\":%zu,\"dof\":%.0f,"
                "\"field_GiB\":%.4f,\"n_md\":%d,\"ms_per_traj\":%.4f,\"traj_per_s\":%.2f,"
                "\"us_force\":%.3f,\"us_sfull\":%.3f,\"us_axpy\":%.3f,\"us_sample\":%.3f}\n",
                label,
                shape.size(),
                shape[0],
                v,
                dof,
                gib,
                n_md,
                t * 1e3,
                1.0 / t,
                us_force,
                us_sfull,
                us_axpy,
                us_sample);
}

template <class Fn>
double time_us_kernel(Fn&& fn) {
    cudaEvent_t a = nullptr;
    cudaEvent_t b = nullptr;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    fn();  // warm
    cudaDeviceSynchronize();
    constexpr int reps = 50;
    cudaEventRecord(a);
    for (int i = 0; i < reps; ++i) {
        fn();
    }
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0.0F;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return (1e3 * static_cast<double>(ms)) / reps;
}

// Lever 1 occupancy sweep: time the SU(3) force + drift kernels at several
// __launch_bounds__(MaxT, MinB) configs in one binary, reporting numRegs and the
// per-thread local-memory spill from cudaFuncGetAttributes (so we see occupancy
// vs. spill without ncu, which is blocked on the managed host). blocks/SM ≈
// 65536 / (numRegs · MaxT); a higher MinB caps regs to raise it.
template <int MaxT, int MinB, class Field>
void su3_lb_config(Field& field, Field& force, Field& mom, std::size_t v, int /*iters*/) {
    using GD                          = reticolo::cuda::SU3Device;
    reticolo::cuda::DeviceTopology const& topo = field.topology();
    cudaFuncAttributes fa{};
    cudaFuncAttributes da{};
    cudaFuncGetAttributes(&fa, reticolo::cuda::su_plaq_force_kernel<GD, MaxT, MinB>);
    cudaFuncGetAttributes(&da, reticolo::cuda::su_expi_lmul_kernel<GD, MaxT, MinB>);
    double const us_force = time_us_kernel([&] {
        reticolo::cuda::su_plaq_force_launch<GD, MaxT, MinB>(
            field.data(), force.data(), topo, -2.0, nullptr);
    });
    double const us_drift = time_us_kernel([&] {
        reticolo::cuda::su_expi_lmul_launch<GD, MaxT, MinB>(
            field.data(), mom.data(), topo, 0.1, nullptr);
    });
    // P100: 65536 regs/SM, 2048 threads/SM. blocks/SM = min(register-limited,
    // thread-limited); occupancy = blocks·MaxT / 2048.
    int const reg_blocks    = 65536 / (fa.numRegs * MaxT);
    int const thread_blocks = 2048 / MaxT;
    int const blocks_per_sm = reg_blocks < thread_blocks ? reg_blocks : thread_blocks;
    double const f_occ      = 100.0 * blocks_per_sm * MaxT / 2048.0;
    std::printf("{\"action\":\"su3\",\"mode\":\"lb_sweep\",\"V\":%zu,\"MaxT\":%d,\"MinB\":%d,"
                "\"force_regs\":%d,\"force_spill\":%d,\"drift_regs\":%d,\"drift_spill\":%d,"
                "\"force_blocks_per_sm\":%d,\"force_occ_pct\":%.1f,"
                "\"us_force\":%.3f,\"us_drift\":%.3f}\n",
                v,
                MaxT,
                MinB,
                fa.numRegs,
                static_cast<int>(fa.localSizeBytes),
                da.numRegs,
                static_cast<int>(da.localSizeBytes),
                blocks_per_sm,
                f_occ,
                us_force,
                us_drift);
}

template <class Field>
void su3_lb_sweep(std::vector<std::size_t> const& shape, int iters) {
    Field field{shape};
    Field force{field.topology()};
    Field mom{field.topology()};
    std::vector<double> const zero(field.size(), 0.0);
    field.copy_from_host(zero.data());
    cudaDeviceSynchronize();
    std::size_t v = 1;
    for (std::size_t s : shape) {
        v *= s;
    }
    su3_lb_config<256, 0>(field, force, mom, v, iters);
    su3_lb_config<256, 2>(field, force, mom, v, iters);
    su3_lb_config<256, 3>(field, force, mom, v, iters);
    su3_lb_config<256, 4>(field, force, mom, v, iters);
    su3_lb_config<128, 4>(field, force, mom, v, iters);
    su3_lb_config<128, 6>(field, force, mom, v, iters);
    su3_lb_config<128, 8>(field, force, mom, v, iters);
}

int arg_int(int argc, char** argv, char const* key, int fallback) {
    std::size_t const klen = std::strlen(key);
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], key, klen) == 0) {
            return std::atoi(argv[i] + klen);
        }
    }
    return fallback;
}

}  // namespace

int main(int argc, char** argv) {
    reticolo::log::off();

    std::string action = "phi4";
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--action=", 9) == 0) {
            action = argv[i] + 9;
        }
    }
    int const ndim        = arg_int(argc, argv, "--ndim=", 4);
    int const L           = arg_int(argc, argv, "--size=", 16);
    int const n_md        = arg_int(argc, argv, "--n_md=", 10);
    int const iters       = arg_int(argc, argv, "--iters=", 30);
    auto const has_flag = [&](char const* f) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], f) == 0) {
                return true;
            }
        }
        return false;
    };
    bool const force_only = has_flag("--force-only");
    bool const lb_sweep   = has_flag("--lb-sweep");

    std::vector<std::size_t> const shape(static_cast<std::size_t>(ndim),
                                         static_cast<std::size_t>(L));

    if (lb_sweep) {
        using G = reticolo::gauge_group::SU3;
        su3_lb_sweep<DeviceField<double, MatrixLayout<G>>>(shape, iters);
        return 0;
    }

    if (action == "phi4") {
        run_config<act::Phi4<double>, DeviceField<double>>(
            "phi4", shape, {.kappa = 0.18, .lambda = 1.0}, n_md, iters, force_only);
    } else if (action == "su3") {
        using G = reticolo::gauge_group::SU3;
        run_config<act::Wilson<G, double>, DeviceField<double, MatrixLayout<G>>>(
            "su3", shape, {.beta = 6.0}, n_md, iters, force_only);
    } else {
        std::fprintf(stderr, "unknown --action=%s (expected phi4|su3)\n", action.c_str());
        return 2;
    }
    return 0;
}
