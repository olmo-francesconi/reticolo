#pragma once

// Device imaginary-part kernels for the relativistic Bose gas — nvcc-only (.cuh;
// launches kernels). The GPU twin of action::BoseGas::s_imag / compute_force_imag.
// Both call the SAME shared RETICOLO_HD formula (bose_gas_formula.hpp) as the CPU
// action — one source of truth for the physics.
//
//   S_I = 2 Σ_x Im(conj(φ_x)·φ_{x+τ})       (forward τ neighbour only)
//   F_I(x) = 2i·(φ_{x+τ} − φ_{x−τ})          (both τ neighbours)
//
// τ is the time direction = last dim (mu = ndim-1). Unlike S_R/F_R these touch
// ONLY the time neighbours, so the kernels are a thin per-site gather of two
// links rather than the full 2d stencil. The S_I reduction reuses the
// deterministic reduce_sum primitive (reduce.cuh) for run-to-run reproducibility.

#include <reticolo/action/complex/formula/bose_gas_formula.hpp>
#include <reticolo/core/cplx.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/reduce.cuh>

#include <cuda_runtime.h>

namespace reticolo::cuda {

// Per-site S_I contribution 2·Im(conj φ_i · φ_{i+τ}) → site_out[i]. Templated for
// weak/mergeable linkage under -rdc=true (as the other header kernels are).
template <class T>
__global__ void bose_s_imag_site_kernel(cplx<T> const* __restrict__ field,
                                        double* __restrict__ site_out,
                                        DeviceTopology topo,
                                        int tau) {
    long const i = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (i >= topo.nsites) {
        return;
    }
    site_out[i] = static_cast<double>(
        action::detail::bose_gas_action_imag_site<T>(field[i], field[topo.next(i, tau)]));
}

// F_I(i) = 2i·(φ_{i+τ} − φ_{i−τ}) → force[i]. Overwrites force (no accumulation).
template <class T>
__global__ void bose_force_imag_kernel(cplx<T> const* __restrict__ field,
                                       cplx<T>* __restrict__ force,
                                       DeviceTopology topo,
                                       int tau) {
    long const i = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (i >= topo.nsites) {
        return;
    }
    force[i] = action::detail::bose_gas_force_imag_site<T>(field[topo.next(i, tau)],
                                                           field[topo.prev(i, tau)]);
}

// S_I to a host double (reduce_sum_f64 finishes on the host). `site_scratch` is a
// device buffer of at least topo.nsites doubles.
template <class T>
[[nodiscard]] inline double bose_s_imag_launch(cplx<T> const* field,
                                               double* site_scratch,
                                               DeviceTopology const& topo,
                                               cudaStream_t stream = nullptr) {
    constexpr int kBlock = 256;
    auto const grid      = static_cast<unsigned>((topo.nsites + kBlock - 1) / kBlock);
    bose_s_imag_site_kernel<T>
        <<<grid, kBlock, 0, stream>>>(field, site_scratch, topo, topo.ndim - 1);
    RETICOLO_CUDA_CHECK_LAUNCH();
    return reduce_sum_f64(site_scratch, topo.nsites, stream);
}

// Device-scalar S_I for the hot loop: writes S_I to out[0], no host sync / no
// allocation. `site_scratch` is topo.nsites doubles, `partials` is
// k_reduce_max_grid doubles — both caller-owned.
template <class T>
inline void bose_s_imag_into(double* out,
                             cplx<T> const* field,
                             double* site_scratch,
                             double* partials,
                             DeviceTopology const& topo,
                             cudaStream_t stream = nullptr) {
    constexpr int kBlock = 256;
    auto const grid      = static_cast<unsigned>((topo.nsites + kBlock - 1) / kBlock);
    bose_s_imag_site_kernel<T>
        <<<grid, kBlock, 0, stream>>>(field, site_scratch, topo, topo.ndim - 1);
    RETICOLO_CUDA_CHECK_LAUNCH();
    reduce_sum_into(out, site_scratch, topo.nsites, partials, stream);
}

template <class T>
inline void bose_force_imag_launch(cplx<T> const* field,
                                   cplx<T>* force,
                                   DeviceTopology const& topo,
                                   cudaStream_t stream = nullptr) {
    constexpr int kBlock = 256;
    auto const grid      = static_cast<unsigned>((topo.nsites + kBlock - 1) / kBlock);
    bose_force_imag_kernel<T>
        <<<grid, kBlock, 0, stream>>>(field, force, topo, topo.ndim - 1);
    RETICOLO_CUDA_CHECK_LAUNCH();
}

// Fused F_I + S_I: one τ-sweep writes force[i] = F_I(i) AND site_out[i] = the S_I
// contribution. F_I needs φ_{i±τ}; S_I needs φ_i·φ_{i+τ} — the forward neighbour
// φ_{i+τ} is loaded once and reused, so the fused kernel is one gather where the
// two-pass path (compute_force_imag + s_imag_into) does two.
template <class T>
__global__ void bose_force_imag_and_s_imag_kernel(cplx<T> const* __restrict__ field,
                                                  cplx<T>* __restrict__ force,
                                                  double* __restrict__ site_out,
                                                  DeviceTopology topo,
                                                  int tau) {
    long const i = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (i >= topo.nsites) {
        return;
    }
    cplx<T> const phi = field[i];
    cplx<T> const fwd = field[topo.next(i, tau)];
    cplx<T> const bwd = field[topo.prev(i, tau)];
    force[i]    = action::detail::bose_gas_force_imag_site<T>(fwd, bwd);
    site_out[i] = static_cast<double>(action::detail::bose_gas_action_imag_site<T>(phi, fwd));
}

// Writes F_I into force and S_I into out[0] (device scalar) in one field gather.
// `site_scratch` is topo.nsites doubles, `partials` is k_reduce_max_grid doubles.
template <class T>
inline void bose_force_imag_and_s_imag_into(double* out,
                                            cplx<T> const* field,
                                            cplx<T>* force,
                                            double* site_scratch,
                                            double* partials,
                                            DeviceTopology const& topo,
                                            cudaStream_t stream = nullptr) {
    constexpr int kBlock = 256;
    auto const grid      = static_cast<unsigned>((topo.nsites + kBlock - 1) / kBlock);
    bose_force_imag_and_s_imag_kernel<T>
        <<<grid, kBlock, 0, stream>>>(field, force, site_scratch, topo, topo.ndim - 1);
    RETICOLO_CUDA_CHECK_LAUNCH();
    reduce_sum_into(out, site_scratch, topo.nsites, partials, stream);
}

}  // namespace reticolo::cuda
