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

namespace reticolo::exec {

// A pass is worth threading only when it moves at least this many bytes — below it,
// fork/join dominates. Calibrated on the Linux/libgomp 32-core SPR node: the
// threaded-beats-serial break-even for the force is ~400-500 KB for BOTH the
// memory-bound Phi4 (8 B/site) and the compute-bound SU(3) (576 B/site), because a
// heavy action has proportionally more bytes/site — so one byte threshold tracks
// both regimes. This replaces the old fixed 16384-site gate, which was wrong both
// ways: too eager for scalars (net loss under ~512 KB), far too lazy for gauge
// (9 MB before SU(3) threaded).
inline constexpr std::size_t k_thread_min_bytes = 512UL * 1024;

// Target working-set per flat-range work item. Small enough that a big lattice
// yields many chunks (load balance across the team), large enough to amortise the
// per-chunk dispatch and keep the inner loop vectorising.
inline constexpr std::size_t k_chunk_bytes = 64UL * 1024;

// Should a pass over `nsites` items of `bytes_per_site` each be threaded?
[[nodiscard]] inline bool want_threads(std::size_t nsites, std::size_t bytes_per_site) noexcept {
    return nsites * bytes_per_site >= k_thread_min_bytes;
}

// Sites per flat-range chunk for a field of `bytes_per_site`, aligned to `gran`
// (the kernel's batch width) and at least one gran-block.
[[nodiscard]] inline std::size_t chunk_for(std::size_t bytes_per_site, std::size_t gran) noexcept {
    std::size_t const per = bytes_per_site != 0 ? bytes_per_site : 1;
    std::size_t const c   = (k_chunk_bytes / per / gran) * gran;  // gran-aligned, ≤ target
    return c != 0 ? c : gran;
}

// Set true (per thread) only inside a reticolo-owned parallel region; the traverse
// leaves worksplit via `omp for` when it holds and run serial otherwise. A mutable
// thread-local by design — it is the region marker the composability rests on.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
inline thread_local bool g_in_traverse_region = false;

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

// Range-convenience: map `worker(base, cnt)` over gran-aligned chunks of [0, n).
// Threshold (want) and chunk size are both derived from `bytes_per_site` — the
// field's per-site footprint — so a heavy field (gauge) threads early and chunks
// small while a light one (scalar) threads later and chunks large. `gran` is the
// kernel's batch width; chunk alignment keeps batched kernels bit-identical to a
// serial full-range sweep. (Only the final chunk carries the remainder.)
template <class Worker>
inline void parallel_map_ranges(std::size_t n,
                                std::size_t bytes_per_site,
                                std::size_t gran,
                                Worker const& worker) {
    std::size_t const chunk   = chunk_for(bytes_per_site, gran);
    std::size_t const n_items = (n + chunk - 1) / chunk;
    parallel_map(want_threads(n, bytes_per_site), n_items, [&](std::size_t i) {
        std::size_t const base = i * chunk;
        worker(base, std::min(chunk, n - base));
    });
}

// Range-convenience: reduce `worker(base, cnt) -> Acc` over gran-aligned chunks of
// [0, n), folded in canonical chunk order → thread-count invariant.
template <class Acc = double, class Worker>
[[nodiscard]] inline Acc parallel_reduce_ranges(std::size_t n,
                                                std::size_t bytes_per_site,
                                                std::size_t gran,
                                                Worker const& worker) {
    std::size_t const chunk   = chunk_for(bytes_per_site, gran);
    std::size_t const n_items = (n + chunk - 1) / chunk;
    return parallel_reduce<Acc>(want_threads(n, bytes_per_site), n_items, [&](std::size_t i) {
        std::size_t const base = i * chunk;
        return worker(base, std::min(chunk, n - base));
    });
}

// ---------- canonical field partition -----------------------------------------
//
// THE decomposition of a field's site range. Every per-site pass — elementwise
// map, reduce, stencil sweep, gauge plane kernel — splits into these items, so
// `schedule(static)`'s contiguous block assignment hands each thread the SAME
// contiguous memory span in every operation of a trajectory: producer→consumer
// locality holds by construction, not by per-call-site convention.
//
// Items are contiguous flat spans made by splitting the OUTERMOST dims (never
// dim 0 — x carries the vectorised streams) until at least
// k_min_partition_items exist: enough items for any thread count and for load
// balance, while a thread's static block stays one contiguous slab. The rule is
// a pure function of the shape — no cache constant, no thread count — so the
// reduce fold order (per-item partials in canonical item order) is
// thread-count- AND machine-independent. 1D lattices fall back to fixed
// 8192-site chunks (only there the final item is ragged).
inline constexpr std::size_t k_min_partition_items = 64;

struct Partition {
    std::size_t n_items;
    std::size_t item_sites;
    std::size_t n_sites;
    std::size_t split_dim;  // items enumerate dims [split_dim, ndims); 0 for 1D
};

template <class Field>
[[nodiscard]] inline Partition partition(Field const& f) noexcept {
    std::size_t const n = f.nsites();
    std::size_t const d = f.ndims();
    if (d <= 1) {
        constexpr std::size_t k_item = 8192;
        return {(n + k_item - 1) / k_item, k_item, n, 0};
    }
    auto const& sh    = f.shape();
    std::size_t items = 1;
    std::size_t k     = d;
    while (k > 1 && items < k_min_partition_items) {
        --k;
        items *= sh[k];
    }
    return {items, n / items, n, k};
}

// Fixed 8-lane accumulation for reduce workers: `term(i)` summed over
// [base, base+cnt) into 8 independent lanes + a scalar tail, combined in a
// fixed pairwise order. Breaks the serial FP dependency chain (a plain
// `s += term(i)` runs at ~1 add/cycle latency, ~10× under memory speed) and
// vectorizes on gcc AND clang without pragmas; unlike pragma reassociation the
// summation order is written in code, so the bits are identical on every
// compiler and ISA width. 8 = one AVX-512 register / a whole multiple of
// NEON/AVX2 — the natural lane count for doubles.
template <class Term>
[[nodiscard]] inline double lane_sum8(std::size_t base, std::size_t cnt, Term const& term) {
    double lane[8]        = {};
    std::size_t const end = base + cnt;
    std::size_t i         = base;
    for (; i + 8 <= end; i += 8) {
        for (std::size_t w = 0; w < 8; ++w) {
            lane[w] += term(i + w);
        }
    }
    double tail = 0.0;
    for (; i < end; ++i) {
        tail += term(i);
    }
    return (((lane[0] + lane[1]) + (lane[2] + lane[3])) +
            ((lane[4] + lane[5]) + (lane[6] + lane[7]))) +
           tail;
}

// Field-convenience: the ONLY way any code sweeps a field's sites. Threshold
// (want_threads), the canonical partition, and determinism live here; the
// caller supplies just the per-site-range work. `worker(base, cnt)` must be
// write-disjoint over the site range for the map form; the reduce folds
// per-item partials in canonical item order (thread-count invariant).
//
// `gran` is the kernel's SIMD batch width, kept for documentation and the
// batched kernels' tail handling: the partition itself ignores it (there is ONE
// partition per field, shared by every op), but item sizes are whole products
// of inner dims, so for even lattice extents every item is a whole number of
// any power-of-two batch — batched kernels split vector/scalar tails per item,
// deterministically.
template <class Field, class Worker>
inline void field_visit(Field const& field, std::size_t /*gran*/, Worker const& worker) {
    Partition const p = partition(field);
    parallel_map(want_threads(p.n_sites, field.bytes_per_site()), p.n_items, [&](std::size_t i) {
        std::size_t const base = i * p.item_sites;
        worker(base, std::min(p.item_sites, p.n_sites - base));
    });
}

template <class Acc = double, class Field, class Worker>
[[nodiscard]] inline Acc
field_reduce(Field const& field, std::size_t /*gran*/, Worker const& worker) {
    Partition const p = partition(field);
    return parallel_reduce<Acc>(
        want_threads(p.n_sites, field.bytes_per_site()), p.n_items, [&](std::size_t i) {
            std::size_t const base = i * p.item_sites;
            return worker(base, std::min(p.item_sites, p.n_sites - base));
        });
}

}  // namespace reticolo::exec
