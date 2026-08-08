#pragma once

// Compact-U(1) device kernels — nvcc-only (.cuh; launches kernels).
//
// The gauge counterpart of stencil.cuh / reduce_fwd.cuh: a thread-per-LINK force
// (a gather, never the CPU scatter) and a thread-per-SITE plaquette energy. The
// link field is direction-major (link (mu, x) at flat index mu·nsites + x), the
// same order as the host MatrixLinkLattice<U1,T> (n_real_components == 1).

#include <reticolo/action/gauge/formula/wilson_u1_formula.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/checkerboard.cuh>
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
__global__ void plaq_energy_site_kernel(T const* __restrict__ field,
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
            double const plaq = action::formula::u1_plaq<double>(
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
__global__ void plaq_force_gather_kernel(T const* __restrict__ field,
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
        double const fwd     = action::formula::u1_plaq<double>(
            static_cast<double>(field[(static_cast<long>(mu) * ns) + x]),
            static_cast<double>(field[(static_cast<long>(nu) * ns) + x_pmu]),
            static_cast<double>(field[(static_cast<long>(mu) * ns) + x_pnu]),
            static_cast<double>(field[(static_cast<long>(nu) * ns) + x]));
        double const bwd = action::formula::u1_plaq<double>(
            static_cast<double>(field[(static_cast<long>(mu) * ns) + x_mnu]),
            static_cast<double>(field[(static_cast<long>(nu) * ns) + x_mnu_pmu]),
            static_cast<double>(field[(static_cast<long>(mu) * ns) + x]),
            static_cast<double>(field[(static_cast<long>(nu) * ns) + x_mnu]));
        acc += sin(fwd) - sin(bwd);
    }
    force[(static_cast<long>(mu) * ns) + x] = static_cast<T>(-beta * acc);
}

// Per-link U(1) Metropolis, one (direction, site parity) colour. Same colouring
// argument as the SU(N) kernel; the staple is a plain complex sum here because a
// 1x1 unitary is just e^{iθ}, so the local action is −β·Re(e^{iθ}·A) with
// A = Σ_ν e^{i·(staple phase)}.
template <class T>
__global__ void u1_metropolis_kernel(T* __restrict__ field,
                                     DeviceTopology topo,
                                     int mu,
                                     int parity,
                                     double sigma,
                                     double beta,
                                     std::uint64_t seed,
                                     std::uint64_t const* __restrict__ sweep,
                                     double* __restrict__ ds_out,
                                     double* __restrict__ acc_out) {
    long const x  = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    long const ns = topo.nsites;
    int const d   = topo.ndim;
    if (x >= ns) {
        return;
    }
    if (site_parity(topo, x) != parity) {
        ds_out[x]  = 0.0;
        acc_out[x] = 0.0;
        return;
    }
    long const x_pmu = topo.next(x, mu);

    // A = Σ_ν [ e^{i(θ_ν(x+μ) − θ_μ(x+ν) − θ_ν(x))}
    //         + e^{i(−θ_ν(x+μ−ν) − θ_μ(x−ν) + θ_ν(x−ν))} ]
    double a_re = 0.0;
    double a_im = 0.0;
    for (int nu = 0; nu < d; ++nu) {
        if (nu == mu) {
            continue;
        }
        long const x_pnu     = topo.next(x, nu);
        long const x_mnu     = topo.prev(x, nu);
        long const x_pmu_mnu = topo.prev(x_pmu, nu);
        auto const th        = [&](int dir, long site) {
            return static_cast<double>(field[(static_cast<long>(dir) * ns) + site]);
        };
        double const fwd = th(nu, x_pmu) - th(mu, x_pnu) - th(nu, x);
        double const bwd = -th(nu, x_pmu_mnu) - th(mu, x_mnu) + th(nu, x_mnu);
        a_re += cos(fwd) + cos(bwd);
        a_im += sin(fwd) + sin(bwd);
    }

    long const link = (static_cast<long>(mu) * ns) + x;
    double n0       = 0.0;
    double n1       = 0.0;
    double u0       = 0.0;
    double u1       = 0.0;
    philox_normal2(seed, *sweep, static_cast<std::uint64_t>(2 * link), n0, n1);
    philox_uniform2(seed, *sweep, static_cast<std::uint64_t>((2 * link) + 1), u0, u1);

    auto const theta  = static_cast<double>(field[link]);
    double const prop = theta + (sigma * n0);
    // S_local = −β·Re(e^{iθ}·A)
    double const s_old = -beta * ((cos(theta) * a_re) - (sin(theta) * a_im));
    double const s_new = -beta * ((cos(prop) * a_re) - (sin(prop) * a_im));
    double const ds    = s_new - s_old;

    // ::log — unqualified `log` resolves to the reticolo::log NAMESPACE here.
    bool const accept = -ds >= ::log(u0);
    if (accept) {
        field[link] = static_cast<T>(prop);
    }
    ds_out[x]  = accept ? ds : 0.0;
    acc_out[x] = accept ? 1.0 : 0.0;
}

template <class T>
void u1_metropolis_launch(T* field,
                          DeviceTopology const& topo,
                          int color,
                          double sigma,
                          double beta,
                          std::uint64_t seed,
                          std::uint64_t const* sweep,
                          double* ds_out,
                          double* acc_out,
                          cudaStream_t stream) {
    int const mu          = color / 2;
    int const parity      = color % 2;
    constexpr int k_block = 256;
    auto const grid       = static_cast<unsigned>((topo.nsites + k_block - 1) / k_block);
    u1_metropolis_kernel<T><<<grid, k_block, 0, stream>>>(
        field, topo, mu, parity, sigma, beta, seed, sweep, ds_out, acc_out);
    RETICOLO_CUDA_CHECK_LAUNCH();
}

// Fused per-link force + per-link plaquette-energy partial, one gather. The LLR
// windowed force needs S on every MD step; the force gather already forms each
// forward plaquette angle Q_nu(x), so we accumulate that plaquette's Wilson
// energy here instead of a separate plaq_energy pass. Each plaquette {mu<nu} at x
// is counted once — at its lower-direction link (mu). Σ over links = S_W.
template <class T>
__global__ void plaq_fused_force_energy_kernel(T const* __restrict__ field,
                                               T* __restrict__ force,
                                               double* __restrict__ energy_partial,
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
    double acc   = 0.0;  // force gather
    double e     = 0.0;  // Σ_{nu>mu} (1 − cos Q_nu(x)) — plaquettes this link owns
    for (int nu = 0; nu < d; ++nu) {
        if (nu == mu) {
            continue;
        }
        long const x_pmu     = topo.next(x, mu);
        long const x_pnu     = topo.next(x, nu);
        long const x_mnu     = topo.prev(x, nu);
        long const x_mnu_pmu = topo.next(x_mnu, mu);
        double const fwd     = action::formula::u1_plaq<double>(
            static_cast<double>(field[(static_cast<long>(mu) * ns) + x]),
            static_cast<double>(field[(static_cast<long>(nu) * ns) + x_pmu]),
            static_cast<double>(field[(static_cast<long>(mu) * ns) + x_pnu]),
            static_cast<double>(field[(static_cast<long>(nu) * ns) + x]));
        double const bwd = action::formula::u1_plaq<double>(
            static_cast<double>(field[(static_cast<long>(mu) * ns) + x_mnu]),
            static_cast<double>(field[(static_cast<long>(nu) * ns) + x_mnu_pmu]),
            static_cast<double>(field[(static_cast<long>(mu) * ns) + x]),
            static_cast<double>(field[(static_cast<long>(nu) * ns) + x_mnu]));
        acc += sin(fwd) - sin(bwd);
        if (nu > mu) {
            e += 1.0 - cos(fwd);
        }
    }
    force[(static_cast<long>(mu) * ns) + x]          = static_cast<T>(-beta * acc);
    energy_partial[(static_cast<long>(mu) * ns) + x] = beta * e;
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

// Force + per-link energy partial in one launch. `energy_partial` is ndim·nsites
// doubles (caller-owned); Σ over it is S_W.
template <class T>
void plaq_fused_launch(T const* field,
                       T* force,
                       double* energy_partial,
                       DeviceTopology const& topo,
                       double beta,
                       cudaStream_t stream) {
    constexpr int kBlock = 256;
    long const total     = topo.nsites * topo.ndim;
    auto const grid      = static_cast<unsigned>((total + kBlock - 1) / kBlock);
    plaq_fused_force_energy_kernel<T>
        <<<grid, kBlock, 0, stream>>>(field, force, energy_partial, topo, beta);
    RETICOLO_CUDA_CHECK_LAUNCH();
}

}  // namespace reticolo::cuda
