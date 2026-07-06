#pragma once

#include <algorithm>
#include <cstddef>
#include <utility>
#include <vector>

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

// Contiguous, `gran`-aligned sub-range [lo, hi) of [0, n) for the calling thread
// within a reticolo-owned region (SPMD partition). `gran` alignment keeps every
// non-final chunk on a batch boundary, so batched kernels give the same per-site
// path — and hence bit-identical results — as the serial full-range sweep; only
// the final chunk carries the [0, gran) global remainder. `hi - lo` may be 0 when
// there are more threads than blocks. Outside OpenMP (or serial) returns [0, n).
[[nodiscard]] inline std::pair<std::size_t, std::size_t>
spmd_chunk(std::size_t n, [[maybe_unused]] std::size_t gran) noexcept {
#ifdef _OPENMP
    std::size_t const nblk = (n + gran - 1) / gran;
    auto const nt          = static_cast<std::size_t>(omp_get_num_threads());
    auto const tid         = static_cast<std::size_t>(omp_get_thread_num());
    std::size_t const lo   = ((nblk * tid) / nt) * gran;
    std::size_t const hi   = std::min(((nblk * (tid + 1)) / nt) * gran, n);
    return {lo, hi};
#else
    return {std::size_t{0}, n};
#endif
}

// Write-disjoint pass: run `worker(base, cnt)` over one `gran`-aligned contiguous
// chunk of [0, n) per thread (SPMD). For batched kernels the alignment makes each
// non-final chunk take the same per-site path as the serial full-range sweep, so
// a write-once pass is bit-identical for any thread count. Standalone opens its
// own region; nested worksplits the current team; foreign/too-small → serial.
template <class Worker>
inline void apply_chunked(std::size_t n, std::size_t gran, Worker const& worker) {
    in_traverse_region(traverse_want(n), [&] {
        if (g_in_traverse_region) {
            auto const [lo, hi] = spmd_chunk(n, gran);
            if (hi > lo) {
                worker(lo, hi - lo);
            }
        } else {
            worker(std::size_t{0}, n);
        }
    });
}

// Deterministic blocked reduction: sum `worker(base, cnt) -> double` over a fixed
// `gran`-block partition of [0, n) into a partials buffer, folded in block order.
// The partition is independent of thread count, so the result is identical for
// any number of threads (the reduction analogue of spmd_chunk). Standalone opens
// its own region; nested worksplits the current team; foreign/too-small → serial.
template <class Worker>
[[nodiscard]] inline double reduce_blocks(std::size_t n, std::size_t gran, Worker const& worker) {
    std::size_t const nblk = (n + gran - 1) / gran;
    std::vector<double> partials(nblk, 0.0);
    in_traverse_region(traverse_want(n), [&] {
        auto block = [&](std::size_t b) {
            std::size_t const lo = b * gran;
            std::size_t const hi = std::min(lo + gran, n);
            partials[b]          = worker(lo, hi - lo);
        };
        if (g_in_traverse_region) {
#pragma omp for schedule(static)
            for (std::ptrdiff_t b = 0; b < static_cast<std::ptrdiff_t>(nblk); ++b) {
                block(static_cast<std::size_t>(b));
            }
        } else {
            for (std::size_t b = 0; b < nblk; ++b) {
                block(b);
            }
        }
    });
    double total = 0.0;
    for (double const v : partials) {
        total += v;
    }
    return total;
}

}  // namespace reticolo::detail
