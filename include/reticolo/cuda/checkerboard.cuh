#pragma once

// Generic scalar CHECKERBOARD skeleton — nvcc-only (.cuh; included from .cu).
//
// The local-update twin of stencil.cuh, and the same shape: one __global__
// template parameterized by an action functor, thread-per-site, functor copied
// into registers, all 2d neighbours streamed through accumulate(). It differs in
// exactly two places — the thread updates the site IN PLACE, and it is masked to
// one colour so the sites it writes are never the sites another thread reads.
//
// Like stencil.cuh it is driven by an ACTION (through its device_functors trait);
// cuda::Metropolis never includes it. Same layering as the host side.
//
// FUNCTOR PROTOCOL (the local-update counterpart of init/accumulate/finalize):
//     init(self)              start the site
//     accumulate(mu, nbr)     stream one neighbour
//     ds(old, prop) -> double S(prop) − S(old) for this site
// A site action's `ds` reads a running neighbour sum; a bond action's keeps the
// 2d raw neighbours in registers, because its per-bond term reads the candidate.
// Either way `ds` binds the SAME RETICOLO_HD formula the host family op binds —
// one source of truth per action, as with the force.
//
// RNG is counter-based Philox keyed on (seed, sweep counter, site) — stateless,
// so there is no per-site stream to store or checkpoint, and the sweep counter is
// read from a DEVICE pointer so a captured graph replays with fresh randomness.
// Two draws per hit: the proposal normal and the accept uniform. As on the CPU
// the uniform is drawn unconditionally and compared as −ΔS ≥ log u, so the draw
// count is a pure function of the shape.
//
// Colour masking is the simple form: launch nsites threads and return on the
// wrong parity, leaving half of every warp idle. The compacted nsites/2 index map
// is the obvious follow-up — the neighbour gather is strided either way, so it
// wants measuring before it is worth the index arithmetic.

#include <reticolo/core/rng/philox.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_topology.hpp>

#include <cstdint>

#include <cuda_runtime.h>

namespace reticolo::cuda {

// Is the field element complex (a `cplx<T>`)? A complex field's proposal moves
// BOTH components at once, so it consumes both normals of the Philox pair rather
// than throwing one away — the device twin of the host's complex Gaussian fill.
template <class T>
concept ComplexElement = requires(T t) {
    t.re;
    t.im;
};

// The proposal offset σ·N for one element: σ·g0 for a real element, σ·(g0 + i·g1)
// for a complex one.
template <class T>
[[nodiscard]] RETICOLO_HD inline T metropolis_offset(double sigma, double g0, double g1) {
    if constexpr (ComplexElement<T>) {
        using R = decltype(T{}.re);
        return T{static_cast<R>(sigma * g0), static_cast<R>(sigma * g1)};
    } else {
        return static_cast<T>(sigma * g0);
    }
}

// Site parity (Σ_mu c_mu) & 1 from the flat index, via the same closed-form
// coordinate decomposition DeviceTopology::next/prev use.
[[nodiscard]] RETICOLO_HD inline int site_parity(DeviceTopology const& topo, long s) {
    long par = 0;
    for (int mu = 0; mu < topo.ndim; ++mu) {
        par += (s / topo.stride[mu]) % topo.shape[mu];
    }
    return static_cast<int>(par & 1);
}

// One colour of one sweep. `ds_out` / `acc_out` are per-site: the applied ΔS
// (0 on reject) and the accepted-hit count, summed afterwards by the ordinary
// reduce_sum_into so this kernel needs no atomics and no block reduction.
template <class F>
__global__ void checkerboard_kernel(F f,
                                    typename F::element* __restrict__ field,
                                    DeviceTopology topo,
                                    int color,
                                    double sigma,
                                    std::uint64_t seed,
                                    std::uint64_t const* sweep,
                                    double* __restrict__ ds_out,
                                    double* __restrict__ acc_out) {
    long const i = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (i >= topo.nsites) {
        return;
    }
    if (site_parity(topo, i) != color) {
        ds_out[i]  = 0.0;
        acc_out[i] = 0.0;
        return;
    }

    using T = typename F::element;
    T cur   = field[i];

    // The neighbours are all of the OTHER colour, so they are frozen for the whole
    // of this kernel — one gather scores the proposal.
    f.init(cur);
    for (int mu = 0; mu < topo.ndim; ++mu) {
        f.accumulate(mu, field[topo.next(i, mu)]);
        f.accumulate(mu, field[topo.prev(i, mu)]);
    }

    auto const pair = static_cast<std::uint64_t>(i);
    double g0       = 0.0;
    double g1       = 0.0;
    double u0       = 0.0;
    double u1       = 0.0;
    philox_normal2(seed, *sweep, 2 * pair, g0, g1);
    // A separate key word for the accept uniform: the proposal and the accept
    // test must be independent draws, never two words of one Philox output
    // reused across roles.
    philox_uniform2(seed, *sweep, (2 * pair) + 1, u0, u1);

    T const prop    = cur + metropolis_offset<T>(sigma, g0, g1);
    double const ds = f.ds(cur, prop);
    // −ΔS ≥ log u — the branchless form of `ΔS ≤ 0 or u < exp(−ΔS)`, and the
    // same test the host family op makes. u == 0 gives −inf and accepts, which
    // is what u < exp(−ΔS) does too.
    // ::log — unqualified `log` resolves to the reticolo::log NAMESPACE here.
    bool const accept = -ds >= ::log(u0);

    field[i]   = accept ? prop : cur;
    ds_out[i]  = accept ? ds : 0.0;
    acc_out[i] = accept ? 1.0 : 0.0;
}

template <class F>
void checkerboard_launch(F const& f,
                         typename F::element* field,
                         DeviceTopology const& topo,
                         int color,
                         double sigma,
                         std::uint64_t seed,
                         std::uint64_t const* sweep,
                         double* ds_out,
                         double* acc_out,
                         cudaStream_t stream = nullptr) {
    constexpr int k_block = 256;
    auto const grid       = static_cast<unsigned>((topo.nsites + k_block - 1) / k_block);
    checkerboard_kernel<F>
        <<<grid, k_block, 0, stream>>>(f, field, topo, color, sigma, seed, sweep, ds_out, acc_out);
    RETICOLO_CUDA_CHECK_LAUNCH();
}

}  // namespace reticolo::cuda
