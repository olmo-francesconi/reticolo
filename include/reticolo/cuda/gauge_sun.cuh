#pragma once

// Generic SU(N) Wilson gauge kernels — nvcc-only (.cuh). Templated on a group
// device-traits type GD (SU2Device / SU3Device, supplying nc / n_color / n_gen
// and the register-local matrix ops). One set of kernels serves every matrix
// group; the group plugs in via GD. The link/momentum/force buffers are the
// MatrixLinkLattice layout [ndim][nc][nsites].
//
//   action:  S = β·n_plaq − (β/N)·Σ_p ReTr U_p   (per-site forward planes)
//   force :  F_μ(x) = −(β/N)·TA[U_μ(x)·V_μ(x)]    (per-link staple GATHER)
//   drift :  U_μ(x) ← exp(dt·P_μ(x))·U_μ(x)        (group exp, ADL atom)
//   sample:  P = Σ_a h_a T_a, h_a ~ N(0,½)         (Gell-Mann scatter)

#include <reticolo/core/philox.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/gauge/group_device.hpp>
#include <reticolo/cuda/reduce.cuh>
#include <reticolo/cuda/stream.hpp>

#include <cstdint>

#include <cuda_runtime.h>
#include <numbers>

namespace reticolo::cuda {

// Launch geometry for the per-link force/drift kernels. kSuBlock is the block
// size; kSuMinBlocks is the __launch_bounds__ minimum blocks/SM occupancy floor
// (0 = no floor). The kernels are templated on these so the profiler's --lb-sweep
// can measure several configs in one build (cudaFuncGetAttributes regs/spill).
//
// Raising the occupancy floor via __launch_bounds__ is a 2.6–7.7× regression:
// the SU(3) staple genuinely needs ~250 regs/thread; capping registers to fit
// ≥2 blocks/SM spills the 3×3 complex matrices to local memory and the spill
// traffic dwarfs the occupancy gain — the kernel is spill-bound, not occupancy-
// latency-bound. So kMinBlocks stays 0; (256,0) matches the no-launch_bounds
// baseline exactly (271 ms/traj at L=32). A real su3 win is structural (stage
// the staple through shared memory / warp-per-link), not occupancy.
inline constexpr int kSuBlock     = 256;
inline constexpr int kSuMinBlocks = 0;

// Per-site Wilson action contribution over forward planes μ<ν:
//   site_out[x] = Σ_{μ<ν} ( β − (β/N)·ReTr U_{μν}(x) )   →   Σ_x = S_W.
template <class GD>
__global__ void
su_plaq_energy_kernel(double const* field, double* site_out, DeviceTopology topo, double beta) {
    long const x = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (x >= topo.nsites) {
        return;
    }
    long const ns       = topo.nsites;
    int const d         = topo.ndim;
    double const beta_n = beta / static_cast<double>(GD::n_color);
    double acc          = 0.0;
    for (int mu = 0; mu < d; ++mu) {
        long const x_pmu = topo.next(x, mu);
        for (int nu = mu + 1; nu < d; ++nu) {
            long const x_pnu = topo.next(x, nu);
            double a[GD::nc];
            double b[GD::nc];
            double c[GD::nc];
            double dmat[GD::nc];
            double ab[GD::nc];
            double dc[GD::nc];
            GD::load(field, mu, x, ns, a);                      // U_μ(x)
            GD::load(field, nu, x_pmu, ns, b);                  // U_ν(x+μ)
            GD::load(field, mu, x_pnu, ns, c);                  // U_μ(x+ν)
            GD::load(field, nu, x, ns, dmat);                   // U_ν(x)
            GD::mul(ab, a, b);                                  // AB = U_μ(x)·U_ν(x+μ)
            GD::mul(dc, dmat, c);                               // DC = U_ν(x)·U_μ(x+ν)
            acc += beta - (beta_n * GD::retr_mul_adj(ab, dc));  // ReTr(AB·DC†)
        }
    }
    site_out[x] = acc;
}

// Per-link staple force gather. scale = −(β/N); writes F = scale·TA[U·V].
template <class GD, int MaxT = kSuBlock, int MinB = kSuMinBlocks>
__global__ void __launch_bounds__(MaxT, MinB)
su_plaq_force_kernel(double const* field, double* force, DeviceTopology topo, double scale) {
    long const tid   = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    long const ns    = topo.nsites;
    int const d      = topo.ndim;
    long const total = ns * d;
    if (tid >= total) {
        return;
    }
    int const mu     = static_cast<int>(tid / ns);
    long const x     = tid % ns;
    long const x_pmu = topo.next(x, mu);

    double v[GD::nc];
    for (int k = 0; k < GD::nc; ++k) {
        v[k] = 0.0;
    }
    for (int nu = 0; nu < d; ++nu) {
        if (nu == mu) {
            continue;
        }
        long const x_pnu     = topo.next(x, nu);
        long const x_mnu     = topo.prev(x, nu);
        long const x_pmu_mnu = topo.prev(x_pmu, nu);
        double a[GD::nc];
        double b[GD::nc];
        double c[GD::nc];
        double t1[GD::nc];
        double t2[GD::nc];
        // forward staple: U_ν(x+μ)·U_μ(x+ν)†·U_ν(x)†
        GD::load(field, nu, x_pmu, ns, a);
        GD::load(field, mu, x_pnu, ns, b);
        GD::load(field, nu, x, ns, c);
        GD::mul_adj(t1, a, b);
        GD::mul_adj(t2, t1, c);
        for (int k = 0; k < GD::nc; ++k) {
            v[k] += t2[k];
        }
        // backward staple: U_ν(x+μ−ν)†·U_μ(x−ν)†·U_ν(x−ν)
        GD::load(field, nu, x_pmu_mnu, ns, a);
        GD::load(field, mu, x_mnu, ns, b);
        GD::load(field, nu, x_mnu, ns, c);
        GD::adj_mul(t1, b, c);
        GD::adj_mul(t2, a, t1);
        for (int k = 0; k < GD::nc; ++k) {
            v[k] += t2[k];
        }
    }
    double u[GD::nc];
    double uv[GD::nc];
    double ta[GD::nc];
    GD::load(field, mu, x, ns, u);
    GD::mul(uv, u, v);
    GD::traceless_antiherm(ta, uv);
    GD::store_scaled(force, mu, x, ns, ta, scale);
}

// Per-link group exponential drift: U ← exp(dt·P)·U.
template <class GD, int MaxT = kSuBlock, int MinB = kSuMinBlocks>
__global__ void __launch_bounds__(MaxT, MinB)
su_expi_lmul_kernel(double* u, double const* p, DeviceTopology topo, double dt) {
    long const tid   = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    long const ns    = topo.nsites;
    int const d      = topo.ndim;
    long const total = ns * d;
    if (tid >= total) {
        return;
    }
    int const mu = static_cast<int>(tid / ns);
    long const x = tid % ns;
    double pl[GD::nc];
    double ul[GD::nc];
    double vl[GD::nc];
    double un[GD::nc];
    GD::load(p, mu, x, ns, pl);
    GD::load(u, mu, x, ns, ul);
    GD::expi(dt, pl, vl);
    GD::mul(un, vl, ul);  // V · U
    GD::store(u, mu, x, ns, un);
}

// Per-link Gell-Mann momentum sampler: h_a ~ N(0,½), P = Σ_a h_a T_a.
template <class GD>
__global__ void su_sample_algebra_kernel(double* mom,
                                         DeviceTopology topo,
                                         std::uint64_t seed,
                                         std::uint64_t const* traj) {
    long const tid   = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    long const ns    = topo.nsites;
    int const d      = topo.ndim;
    long const total = ns * d;
    if (tid >= total) {
        return;
    }
    int const mu                 = static_cast<int>(tid / ns);
    long const x                 = tid % ns;
    constexpr double k_inv_sqrt2 = std::numbers::sqrt2 / 2.0;
    double h[GD::n_gen];
    // Draw n_gen normals from Philox pairs keyed on (seed, *traj, link·pairs).
    long const pair0 = tid * ((GD::n_gen + 1) / 2);
    for (int a = 0; a < GD::n_gen; a += 2) {
        double n0 = 0.0;
        double n1 = 0.0;
        philox_normal2(seed, *traj, static_cast<std::uint64_t>(pair0 + (a / 2)), n0, n1);
        h[a] = n0 * k_inv_sqrt2;
        if (a + 1 < GD::n_gen) {
            h[a + 1] = n1 * k_inv_sqrt2;
        }
    }
    double pl[GD::nc];
    GD::pack_algebra(h, pl);
    GD::store(mom, mu, x, ns, pl);
}

// ---- launchers ----------------------------------------------------------

template <class GD>
void su_plaq_energy_launch(double const* field,
                           double* site_scratch,
                           DeviceTopology const& topo,
                           double beta,
                           cudaStream_t stream) {
    constexpr int kBlock = 256;
    auto const grid      = static_cast<unsigned>((topo.nsites + kBlock - 1) / kBlock);
    su_plaq_energy_kernel<GD><<<grid, kBlock, 0, stream>>>(field, site_scratch, topo, beta);
    RETICOLO_CUDA_CHECK_LAUNCH();
}

template <class GD, int MaxT = kSuBlock, int MinB = kSuMinBlocks>
void su_plaq_force_launch(double const* field,
                          double* force,
                          DeviceTopology const& topo,
                          double scale,
                          cudaStream_t stream) {
    long const total = topo.nsites * topo.ndim;
    auto const grid  = static_cast<unsigned>((total + MaxT - 1) / MaxT);
    su_plaq_force_kernel<GD, MaxT, MinB><<<grid, MaxT, 0, stream>>>(field, force, topo, scale);
    RETICOLO_CUDA_CHECK_LAUNCH();
}

template <class GD, int MaxT = kSuBlock, int MinB = kSuMinBlocks>
void su_expi_lmul_launch(double* u,
                         double const* p,
                         DeviceTopology const& topo,
                         double dt,
                         cudaStream_t stream) {
    long const total = topo.nsites * topo.ndim;
    auto const grid  = static_cast<unsigned>((total + MaxT - 1) / MaxT);
    su_expi_lmul_kernel<GD, MaxT, MinB><<<grid, MaxT, 0, stream>>>(u, p, topo, dt);
    RETICOLO_CUDA_CHECK_LAUNCH();
}

template <class GD>
void su_sample_algebra_launch(double* mom,
                              DeviceTopology const& topo,
                              std::uint64_t seed,
                              std::uint64_t const* traj,
                              cudaStream_t stream) {
    constexpr int kBlock = 256;
    long const total     = topo.nsites * topo.ndim;
    auto const grid      = static_cast<unsigned>((total + kBlock - 1) / kBlock);
    su_sample_algebra_kernel<GD><<<grid, kBlock, 0, stream>>>(mom, topo, seed, traj);
    RETICOLO_CUDA_CHECK_LAUNCH();
}

// Matrix-field drift atom: U ← exp(cdt·P)·U. Found by ADL when the unchanged
// Integ::run instantiates on a MatrixLayout<G> field; GD resolved from G. The
// kick stays the generic axpy atom (integ_ops.hpp) — the algebra is additive.
template <class G>
inline void drift_field(DeviceField<double, MatrixLayout<G>>& field,
                        DeviceField<double, MatrixLayout<G>> const& mom,
                        double cdt) {
    using GD = typename group_device<G>::type;
    su_expi_lmul_launch<GD>(field.data(), mom.data(), field.topology(), cdt, current_stream());
}

}  // namespace reticolo::cuda
