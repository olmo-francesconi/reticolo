#pragma once

#include <cstddef>

#ifdef _OPENMP
    #include <omp.h>
#endif

// OpenMP composability for the lattice-traverse stage (Model B). A kernel pass
// must thread correctly in three contexts: standalone (a bench calling
// `compute_force`), nested inside the persistent per-trajectory region, and
// nested inside LLR's replica-parallel region (where it must stay serial so it
// doesn't fight the replica team). The mechanism:
//
//   * `in_traverse_region(want, body)` — the entry wrapper. When `want` (the
//     lattice is large enough) and we're not already inside any parallel region,
//     it opens ONE `omp parallel` and marks it ours by setting the thread-local
//     `g_in_traverse_region`. Otherwise it runs `body` inline — nested in our own
//     region the flag is already set (leaves worksplit); inside a foreign region
//     or a too-small pass the flag is false (leaves run serial).
//   * Every LEAF (visit_nn, reduce_fwd, drift, kick, …) branches on
//     `g_in_traverse_region`: true → orphaned `#pragma omp for` binding to the
//     enclosing reticolo region; false → a plain serial loop.
//
// So a trajectory opens the region once and all its kernels worksplit within it
// — the per-call fork/join barriers of the old model collapse to cheap barriers
// on a team that stays hot. Without `-fopenmp` every pragma vanishes and this all
// degrades to serial.

namespace reticolo::detail {

// Per-core working-set target for cache tiling. Fixed on purpose: for the
// bandwidth-bound stencil the block sweet spot is ~L2-size-independent (filling a
// detected L2 overshoots and regresses moderate volumes). ~½ of a common 1 MB L2.
inline constexpr std::size_t k_traverse_l2_bytes = 512UL * 1024;

// Below this site count a pass stays single-threaded (fork/join not amortised).
inline constexpr std::size_t k_traverse_min_sites = 1UL << 14;  // 16384

// Set true (per thread) only inside a reticolo-owned parallel region; the traverse
// leaves worksplit via `omp for` when it holds and run serial otherwise. A mutable
// thread-local by design — it is the region marker the composability rests on.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
inline thread_local bool g_in_traverse_region = false;

// A pass this size wants threads (the caller still opens the region only when not
// already nested — see in_traverse_region).
[[nodiscard]] inline bool traverse_want(std::size_t nsites) noexcept {
    return nsites >= k_traverse_min_sites;
}

// body is run by every thread in the team (SPMD), so it is invoked as a const
// lvalue, never forwarded/moved.
template <class Body>
inline void in_traverse_region([[maybe_unused]] bool want, Body const& body) {
#ifdef _OPENMP
    if (want && omp_in_parallel() == 0) {
#pragma omp parallel
        {
            bool const prev      = g_in_traverse_region;
            g_in_traverse_region = true;
            body();
            g_in_traverse_region = prev;
        }
        return;
    }
#endif
    body();
}

}  // namespace reticolo::detail
