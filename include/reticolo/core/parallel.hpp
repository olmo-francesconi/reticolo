#pragma once

#include <reticolo/core/log.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <mutex>
#include <set>
#include <string>
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

// Target working-set per flat-range work item. Small enough that a big lattice
// yields many chunks (load balance across the team), large enough to amortise the
// per-chunk dispatch and keep the inner loop vectorising.
inline constexpr std::size_t k_chunk_bytes = 64UL * 1024;

// The traverse team size for THIS thread: how many threads the next lattice pass
// opens. 0 = inherit the OpenMP ambient (`omp_get_max_threads()`, i.e. the
// process-wide OMP_NUM_THREADS). An algorithm binds an explicit size for the span
// of its work via `team_scope`, so the thread count is a property of the CALLER,
// not a global gate — which is what lets a future LLR split the budget as
// `m` threads/replica × `max/m` replicas. Read on the thread that DECIDES the
// count (the one about to open the region); the spawned workers never read it.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
inline thread_local int g_team_size = 0;

// RAII: bind the traverse team size for the current scope, restoring the previous
// on exit (nesting-safe). `n <= 0` inherits the ambient. Hmc binds one of these
// for the span of a trajectory so every per-site pass under it slices the same way.
class team_scope {
public:
    explicit team_scope(int n) noexcept : prev_{g_team_size} { g_team_size = n; }
    team_scope(team_scope const&)            = delete;
    team_scope(team_scope&&)                 = delete;
    team_scope& operator=(team_scope const&) = delete;
    team_scope& operator=(team_scope&&)      = delete;
    ~team_scope() noexcept { g_team_size = prev_; }

private:
    int prev_;
};

// How many threads the next lattice pass opens: the caller's bound team size if
// set, else the OpenMP ambient. No byte gate — the user (via team_scope or
// OMP_NUM_THREADS) decides. The team is then capped at the work-item count in
// parallel_map/parallel_reduce, so a pass never spawns more threads than it has
// slabs to hand out. No OpenMP → always 1. Args kept for call-site symmetry.
[[nodiscard]] inline int traverse_threads(std::size_t /*nsites*/,
                                          std::size_t /*bytes_per_site*/) noexcept {
#ifdef _OPENMP
    int const n = g_team_size > 0 ? g_team_size : omp_get_max_threads();
    return n > 1 ? n : 1;
#else
    return 1;
#endif
}

// Sites per flat-range chunk for a field of `bytes_per_site`, aligned to `gran`
// (the kernel's batch width) and at least one gran-block.
[[nodiscard]] inline std::size_t chunk_for(std::size_t bytes_per_site, std::size_t gran) noexcept {
    std::size_t const per = bytes_per_site != 0 ? bytes_per_site : 1;
    std::size_t const c   = (k_chunk_bytes / per / gran) * gran;  // gran-aligned, ≤ target
    return c != 0 ? c : gran;
}

// Grow a caller-owned reusable staging buffer to at least `n` and return its
// data pointer. The batched kernels stage transcendental/scatter lanes in a
// per-thread scratch to split the RNG/compute-bound and memory-bound phases;
// this factors the grow-on-demand idiom they all share. The buffer stays a
// NAMED `thread_local` at each call site (not a single shared slab keyed by
// type) so two same-typed scratches that are live at once — e.g. a sampler's
// draw buffer and the `normal_fill` it calls — can never alias.
template <class T>
inline T* thread_scratch(std::vector<T>& buf, std::size_t n) {
    if (buf.size() < n) {
        buf.resize(n);
    }
    return buf.data();
}

// Set true (per thread) only inside a reticolo-owned parallel region; the traverse
// leaves worksplit via `omp for` when it holds and run serial otherwise. A mutable
// thread-local by design — it is the region marker the composability rests on.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
inline thread_local bool g_in_traverse_region = false;

// body is run by every thread in the team (SPMD), so it is invoked as a const
// lvalue, never forwarded/moved.
template <class Body>
inline void in_traverse_region([[maybe_unused]] int nthreads, Body const& body) {
#ifdef _OPENMP
    if (nthreads > 1 && omp_in_parallel() == 0) {
    #pragma omp parallel num_threads(nthreads)
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
// `nthreads` is the team size (from traverse_threads); 1 runs serial. Standalone
// opens its own region; nested in a reticolo region worksplits the current team.
// Never more threads than items — a bigger team would leave workers idle.

template <class Worker>
inline void parallel_map(int nthreads, std::size_t n_items, Worker const& worker) {
    if (static_cast<std::size_t>(nthreads) > n_items) {
        nthreads = n_items > 0 ? static_cast<int>(n_items) : 1;
    }
    in_traverse_region(nthreads, [&] {
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
[[nodiscard]] inline Acc parallel_reduce(int nthreads, std::size_t n_items, Worker const& worker) {
    if (static_cast<std::size_t>(nthreads) > n_items) {
        nthreads = n_items > 0 ? static_cast<int>(n_items) : 1;
    }
    std::vector<Acc> partials(n_items, Acc{});
    in_traverse_region(nthreads, [&] {
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
    parallel_map(traverse_threads(n, bytes_per_site), n_items, [&](std::size_t i) {
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
    return parallel_reduce<Acc>(traverse_threads(n, bytes_per_site), n_items, [&](std::size_t i) {
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

// Human byte size, e.g. 2048 → "2.0 KiB", for the slab log.
[[nodiscard]] inline std::string human_bytes(std::size_t n) {
    double v         = static_cast<double>(n);
    char const* unit = "B";
    for (char const* u : std::array<char const*, 3>{"KiB", "MiB", "GiB"}) {
        if (v < 1024.0) {
            break;
        }
        v /= 1024.0;
        unit = u;
    }
    return std::format("{:.1f} {}", v, unit);
}

// Log how a field is sliced across the team — the full lattice and one slab's
// geometry (a slab is dims [0, split_dim) FULL × a single index in each outer dim;
// 1D = a flat item_sites chunk) plus its byte footprint, e.g. `16x16x16x16 (65536)
// → 16x16x1x1 (256 sites, 2.0 KiB) slab across 4 threads`. ONE line per distinct
// description, only when it actually threads. `request` is the team asked for
// (traverse_threads); the effective count is capped at the slab count. Deduped so
// the per-op traversal — force, s_full, kick, drift, momentum, copy — logs once.
inline void note_slicing([[maybe_unused]] std::size_t const* shape,
                         [[maybe_unused]] std::size_t ndims,
                         [[maybe_unused]] std::size_t bytes_per_site,
                         [[maybe_unused]] Partition const& p,
                         [[maybe_unused]] int request) {
#ifdef _OPENMP
    int const nthr = request < static_cast<int>(p.n_items) ? request
                                                           : static_cast<int>(p.n_items);
    if (nthr <= 1) {
        return;
    }
    std::string full;
    std::string slab;
    for (std::size_t mu = 0; mu < ndims; ++mu) {
        if (mu != 0) {
            full += 'x';
            slab += 'x';
        }
        full += std::to_string(shape[mu]);
        std::size_t const s = p.split_dim == 0 ? (mu == 0 ? p.item_sites : 1)
                              : mu < p.split_dim ? shape[mu]
                                                 : std::size_t{1};
        slab += std::to_string(s);
    }
    std::string const line = std::format("{} ({}) → {} ({} sites, {}) slab across {} threads",
                                          full, p.n_sites, slab, p.item_sites,
                                          human_bytes(p.item_sites * bytes_per_site), nthr);
    static std::mutex mu_lock;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    static std::set<std::string> seen;
    {
        std::lock_guard<std::mutex> const lk(mu_lock);
        if (!seen.insert(line).second) {
            return;
        }
    }
    reticolo::log::info("slic", "{}", line);
#endif
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

// Field-convenience: the ONLY way any code sweeps a field's sites. The team size
// (traverse_threads), the canonical partition, and determinism live here; the
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
    Partition const p  = partition(field);
    int const nthreads = traverse_threads(p.n_sites, field.bytes_per_site());
    note_slicing(field.shape().data(), field.ndims(), field.bytes_per_site(), p, nthreads);
    parallel_map(nthreads, p.n_items, [&](std::size_t i) {
        std::size_t const base = i * p.item_sites;
        worker(base, std::min(p.item_sites, p.n_sites - base));
    });
}

template <class Acc = double, class Field, class Worker>
[[nodiscard]] inline Acc
field_reduce(Field const& field, std::size_t /*gran*/, Worker const& worker) {
    Partition const p  = partition(field);
    int const nthreads = traverse_threads(p.n_sites, field.bytes_per_site());
    note_slicing(field.shape().data(), field.ndims(), field.bytes_per_site(), p, nthreads);
    return parallel_reduce<Acc>(nthreads, p.n_items, [&](std::size_t i) {
        std::size_t const base = i * p.item_sites;
        return worker(base, std::min(p.item_sites, p.n_sites - base));
    });
}

}  // namespace reticolo::exec
