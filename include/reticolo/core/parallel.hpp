#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

#ifdef _OPENMP
    #include <omp.h>
#endif

// OpenMP composability + the two parallel primitives for the lattice-traverse
// stage. A kernel pass must thread correctly in three contexts: standalone (a
// bench calling `compute_force`), nested inside a per-trajectory region, and
// nested inside LLR's replica-parallel region (where it must stay serial so it
// doesn't fight the replica team). The mechanism:
//
//   * `in_traverse_region(want, body)` — the region entry. When `want` and we're
//     not already inside any parallel region, it opens ONE `omp parallel` and
//     marks it ours via the thread-local `g_in_traverse_region`. Otherwise it runs
//     `body` inline — nested in our own region the flag is already set (worksplit);
//     inside a foreign region or a too-small pass the flag is false (serial).
//   * `parallel_map` / `parallel_reduce` — the ONLY places that hold an
//     `#pragma omp for` and read `g_in_traverse_region`. Kernels are pure workers
//     over a work-item index; they never see OpenMP. See below.
//
// Without `-fopenmp` every pragma vanishes and this all degrades to serial.

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

// The two parallel primitives. Every threaded kernel in the tree is expressed as
// one of these over an abstract work-item count; the `omp for` and the
// `g_in_traverse_region` branch live here and nowhere else. A *work item* is
// whatever the caller makes it — a lattice tile, a stencil row, a site chunk.
//
//  * parallel_map    — write-disjoint. Each item's worker writes its own region,
//                      so the result is order-independent → bit-identical for any
//                      thread count. (force, kick, drift, momentum, copy.)
//  * parallel_reduce — deterministic. The item partition is fixed (independent of
//                      thread count), so one partial per item folded in canonical
//                      item order is identical for any thread count. (s_full,
//                      kinetic.)
//
// `want` gates threading — the caller knows the problem size (traverse_want).
// Standalone opens its own region; nested in a reticolo region worksplits the
// current team; a foreign region or !want runs the serial branch.

template <class Worker>
inline void parallel_map(bool want, std::size_t n_items, Worker const& worker) {
    in_traverse_region(want, [&] {
        if (g_in_traverse_region) {
#pragma omp for schedule(static)
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n_items); ++i) {
                worker(static_cast<std::size_t>(i));
            }
        } else {
            for (std::size_t i = 0; i < n_items; ++i) {
                worker(i);
            }
        }
    });
}

template <class Acc = double, class Worker>
[[nodiscard]] inline Acc parallel_reduce(bool want, std::size_t n_items, Worker const& worker) {
    std::vector<Acc> partials(n_items, Acc{});
    in_traverse_region(want, [&] {
        if (g_in_traverse_region) {
#pragma omp for schedule(static)
            for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(n_items); ++i) {
                partials[static_cast<std::size_t>(i)] = worker(static_cast<std::size_t>(i));
            }
        } else {
            for (std::size_t i = 0; i < n_items; ++i) {
                partials[i] = worker(i);
            }
        }
    });
    Acc total{};
    for (Acc const& v : partials) {
        total += v;
    }
    return total;
}

// Range-convenience: map `worker(base, cnt)` over `chunk`-sized, chunk-aligned
// pieces of [0, n). The caller picks the granularity — coarse for a heavy per-site
// pipeline (the SU(N) drift amortises its multi-pass scratch), fine for cheap
// elementwise ops. Chunk alignment keeps batched kernels bit-identical to a serial
// full-range sweep. (Only the final chunk carries the [0, chunk) remainder.)
template <class Worker>
inline void parallel_map_ranges(bool want, std::size_t n, std::size_t chunk, Worker const& worker) {
    std::size_t const n_items = (n + chunk - 1) / chunk;
    parallel_map(want, n_items, [&](std::size_t i) {
        std::size_t const base = i * chunk;
        worker(base, std::min(chunk, n - base));
    });
}

// Range-convenience: reduce `worker(base, cnt) -> Acc` over chunk-aligned pieces of
// [0, n), folded in canonical chunk order → thread-count invariant.
template <class Acc = double, class Worker>
[[nodiscard]] inline Acc
parallel_reduce_ranges(bool want, std::size_t n, std::size_t chunk, Worker const& worker) {
    std::size_t const n_items = (n + chunk - 1) / chunk;
    return parallel_reduce<Acc>(want, n_items, [&](std::size_t i) {
        std::size_t const base = i * chunk;
        return worker(base, std::min(chunk, n - base));
    });
}

}  // namespace reticolo::detail
