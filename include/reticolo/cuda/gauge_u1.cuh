#pragma once

// Compact-U(1) device kernels — nvcc-only (.cuh; launches kernels).
//
// The gauge counterpart of stencil.cuh / reduce_fwd.cuh: a thread-per-LINK force
// (a gather, never the CPU scatter) and a thread-per-SITE plaquette energy. The
// link field is direction-major (link (mu, x) at flat index mu·nsites + x), the
// same order as the host LinkLattice<T>.

#include <reticolo/action/detail/gauge/compact_u1_formula.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/reduce.cuh>

#include <cmath>

#include <cuda_runtime.h>

namespace reticolo::cuda {

// Per-site Wilson plaquette action contribution, summed over the forward planes
// mu < nu through site x: site_out[x] = beta · Σ_{mu<nu} (1 − cos θ_{mu,nu}(x)).
// Σ_x site_out[x] is the full action S_W (so the volume sum needs no constant).
// Angles accumulate in double — the float-sum-corrupts-ΔH rule.
template <class T>
__global__ void
plaq_energy_site_kernel(T const* __restrict__ field,
                        double* __restrict__ site_out,
                        DeviceTopology topo,
                        double beta) {
    long const x = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (x >= topo.nsites) {
        return;
    }
    long const ns = topo.nsites;
    int const d   = topo.ndim;
    double acc    = 0.0;
    for (int mu = 0; mu < d; ++mu) {
        long const x_pmu = topo.next(x, mu);
        for (int nu = mu + 1; nu < d; ++nu) {
            long const x_pnu  = topo.next(x, nu);
            double const plaq = action::detail::u1_plaq<double>(
                static_cast<double>(field[(static_cast<long>(mu) * ns) + x]),
                static_cast<double>(field[(static_cast<long>(nu) * ns) + x_pmu]),
                static_cast<double>(field[(static_cast<long>(mu) * ns) + x_pnu]),
                static_cast<double>(field[(static_cast<long>(nu) * ns) + x]));
            acc += 1.0 - cos(plaq);
        }
    }
    site_out[x] = beta * acc;
}

// Per-link force F(mu, x) = -dS/dtheta_mu(x), re-derived as a GATHER over the
// 2(d-1) plaquettes through the link:
//   F = -beta · Σ_{nu≠mu} [ sin Q_nu(x) - sin Q_nu(x-nu) ],
//   Q_nu(y) = theta_mu(y) + theta_nu(y+mu) - theta_mu(y+nu) - theta_nu(y).
// One thread per link (ndim·nsites threads). Reads neighbours, writes only its
// own link — race-free, unlike the CPU scatter.
template <class T>
__global__ void
plaq_force_gather_kernel(T const* __restrict__ field,
                         T* __restrict__ force,
                         DeviceTopology topo,
                         double beta) {
    long const tid   = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    long const ns    = topo.nsites;
    int const d      = topo.ndim;
    long const total = ns * d;
    if (tid >= total) {
        return;
    }
    int const mu = static_cast<int>(tid / ns);
    long const x = tid % ns;
    double acc   = 0.0;
    for (int nu = 0; nu < d; ++nu) {
        if (nu == mu) {
            continue;
        }
        long const x_pmu     = topo.next(x, mu);
        long const x_pnu     = topo.next(x, nu);
        long const x_mnu     = topo.prev(x, nu);
        long const x_mnu_pmu = topo.next(x_mnu, mu);
        double const fwd     = action::detail::u1_plaq<double>(
            static_cast<double>(field[(static_cast<long>(mu) * ns) + x]),
            static_cast<double>(field[(static_cast<long>(nu) * ns) + x_pmu]),
            static_cast<double>(field[(static_cast<long>(mu) * ns) + x_pnu]),
            static_cast<double>(field[(static_cast<long>(nu) * ns) + x]));
        double const bwd = action::detail::u1_plaq<double>(
            static_cast<double>(field[(static_cast<long>(mu) * ns) + x_mnu]),
            static_cast<double>(field[(static_cast<long>(nu) * ns) + x_mnu_pmu]),
            static_cast<double>(field[(static_cast<long>(mu) * ns) + x]),
            static_cast<double>(field[(static_cast<long>(nu) * ns) + x_mnu]));
        acc += sin(fwd) - sin(bwd);
    }
    force[(static_cast<long>(mu) * ns) + x] = static_cast<T>(-beta * acc);
}

// ---- launchers ----------------------------------------------------------

template <class T>
void plaq_energy_launch(T const* field,
                        double* site_scratch,
                        DeviceTopology const& topo,
                        double beta,
                        cudaStream_t stream) {
    constexpr int kBlock = 256;
    auto const grid      = static_cast<unsigned>((topo.nsites + kBlock - 1) / kBlock);
    plaq_energy_site_kernel<T><<<grid, kBlock, 0, stream>>>(field, site_scratch, topo, beta);
    RETICOLO_CUDA_CHECK_LAUNCH();
}

template <class T>
void plaq_force_launch(
    T const* field, T* force, DeviceTopology const& topo, double beta, cudaStream_t stream) {
    constexpr int kBlock = 256;
    long const total     = topo.nsites * topo.ndim;
    auto const grid      = static_cast<unsigned>((total + kBlock - 1) / kBlock);
    plaq_force_gather_kernel<T><<<grid, kBlock, 0, stream>>>(field, force, topo, beta);
    RETICOLO_CUDA_CHECK_LAUNCH();
}

}  // namespace reticolo::cuda
