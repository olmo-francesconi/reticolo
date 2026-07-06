#pragma once

#include <cstddef>

#ifdef _OPENMP
    #include <omp.h>
#endif

// OpenMP glue for the lattice-traverse stage. The traversal helpers
// (action/*/detail/traversal.hpp, plane.hpp) decide at runtime whether to
// spread a single kernel pass across threads via `traverse_parallel(nsites)`.
//
// Two gates, both load-bearing:
//   * a size threshold — below it the fork/join barrier costs more than the
//     kernel saves, and threading only adds variance (the reliability concern);
//   * `!omp_in_parallel()` — inside the LLR replica-parallel region the cores
//     are already saturated, so nested traverse-threading would oversubscribe.
//
// One `#pragma omp parallel for` per kernel call: libomp parks its team between
// calls, so repeated regions cost a warm barrier (~µs), not thread creation.
// Without `-fopenmp` every pragma vanishes and this returns false, so the whole
// stack degrades to the serial path (Apple Clang, bare builds).

namespace reticolo::detail {

// Per-core working-set target for cache tiling. Tiles are sized so tile+halo
// fits here. A fixed value on purpose: for the bandwidth-bound stencil the block
// sweet spot is roughly L2-size-independent (a few hundred KB — good L1
// residency + prefetch runway), so sizing to *fill* a detected L2 was measured
// to overshoot and regress moderate volumes. ~½ of a common 1 MB L2.
inline constexpr std::size_t k_traverse_l2_bytes = 512UL * 1024;

// Below this site count a pass stays single-threaded (fork/join not amortised).
inline constexpr std::size_t k_traverse_min_sites = 1UL << 14;  // 16384

[[nodiscard]] inline bool traverse_parallel(std::size_t nsites) noexcept {
#ifdef _OPENMP
    return nsites >= k_traverse_min_sites && omp_in_parallel() == 0;
#else
    (void)nsites;
    return false;
#endif
}

}  // namespace reticolo::detail
