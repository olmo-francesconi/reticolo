#pragma once

#include <reticolo/core/log.hpp>

#include <algorithm>
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

// Fleet-wide upper bound on the L1 cache-line size (x86 = 64, Apple Silicon =
// 128). Every line-geometry padding in the tree derives from this one constant:
// StreamSet's per-stream false-sharing slots and the gauge link-span set-aliasing
// pad. An upper bound is always correct — over-padding costs bytes, under-padding
// costs line ping-pong / set collisions — and on x86 128 is what you want anyway
// (the adjacent-line prefetcher couples 64 B line pairs). Deliberately NOT
// std::hardware_destructive_interference_size: that is a per-target compile-time
// guess, not a machine detection, its value follows -mtune on GCC (an ODR/ABI
// hazard in an INTERFACE library, see -Winterference-size), and it disagrees
// across this project's CI toolchains (64 / 128 / 256).
inline constexpr std::size_t k_cache_line_bytes = 128;

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

// Slabs each thread gets — the partition granularity knob. `partition()` aims for
// `team × slabs_per_thread` slabs, snapped to the coarsest achievable outer-dim
// block tiling (see below). 0 = the default 1 (one slab per thread: coarsest split,
// least dispatch overhead, perfect balance when the team divides the count).
// Raising it splits finer (smaller slabs, better cache/imbalance tolerance). Since
// the slab count then depends on the team size, s_full/kinetic reduces are
// deterministic for a FIXED (team, slabs_per_thread) but NOT across different team
// sizes — reproducing a chain means holding both. The write-disjoint maps (force,
// kick, …) stay bit-identical regardless. Bound per-scope like team_scope.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
inline thread_local int g_slabs_per_thread = 0;

class slab_scope {
public:
    explicit slab_scope(int n) noexcept : prev_{g_slabs_per_thread} { g_slabs_per_thread = n; }
    slab_scope(slab_scope const&)            = delete;
    slab_scope(slab_scope&&)                 = delete;
    slab_scope& operator=(slab_scope const&) = delete;
    slab_scope& operator=(slab_scope&&)      = delete;
    ~slab_scope() noexcept { g_slabs_per_thread = prev_; }

private:
    int prev_;
};

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
//                      thread count OR slab count. (force, kick, drift, momentum, copy.)
//  * parallel_reduce — deterministic for a fixed partition. Per-item partials folded
//                      in canonical item order → identical for a given (team, slabs)
//                      config; the partition now tracks the team size, so a different
//                      team or slab count re-folds and can differ in the last ULP.
//                      (s_full, kinetic.)
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
// Chunk size derives from `bytes_per_site` — a heavy field (gauge) gets smaller
// chunks that stay in cache, a light one (scalar) larger chunks. The thread count
// is the caller's team size (traverse_threads); there is no byte-gated threshold.
// `gran` is the kernel's batch width; chunk alignment keeps batched kernels
// bit-identical to a serial full-range sweep. (Only the final chunk carries the
// remainder.)
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
// dim 0 — x carries the vectorised streams). Target count = team ×
// slabs_per_thread; a slab is dims [0, split_dim) FULL × a contiguous `block` of
// dim split_dim × a single index in each higher dim, so it stays one contiguous
// span and the tiling is exact (item_sites = n/n_items). Because the count tracks
// the team size, the reduce fold (per-item partials in canonical item order) is
// deterministic for a FIXED (team, slabs_per_thread) but not across team sizes.
// 1D lattices fall back to flat chunks (only there the final item is ragged).
struct Partition {
    std::size_t n_items;
    std::size_t item_sites;
    std::size_t n_sites;
    std::size_t split_dim;  // dims [0,split_dim) full; dim split_dim blocked by `block`;
                            // dims (split_dim,ndims) single-index. 0 for 1D flat chunks.
    std::size_t block;      // contiguous extent on split_dim (1 = single index)
};

// Is a candidate slab count `items` a better occupancy fit than `best` for a team
// of `team`, aiming at `target`? Prefer a multiple of `team` (every thread equally
// loaded); else the nearest count to `target`, ties rounding UP.
[[nodiscard]] inline bool
better_slabs(std::size_t items, std::size_t best, std::size_t target, std::size_t team) noexcept {
    if (best == 0) {
        return true;
    }
    bool const a = items % team == 0;
    bool const b = best % team == 0;
    if (a != b) {
        return a;
    }
    std::size_t const da = items > target ? items - target : target - items;
    std::size_t const db = best > target ? best - target : target - best;
    return da < db || (da == db && items > best);
}

template <class Field>
[[nodiscard]] inline Partition partition(Field const& f) noexcept {
    std::size_t const n = f.nsites();
    std::size_t const d = f.ndims();
    auto const team     = static_cast<std::size_t>(traverse_threads(0, 0));
    std::size_t const q =
        g_slabs_per_thread > 0 ? static_cast<std::size_t>(g_slabs_per_thread) : std::size_t{1};
    std::size_t const target = team * q;

    if (d <= 1) {
        std::size_t const items = std::clamp<std::size_t>(target, 1, n != 0 ? n : 1);
        std::size_t const chunk = (n + items - 1) / items;
        return {
            .n_items = items, .item_sites = chunk, .n_sites = n, .split_dim = 0, .block = chunk};
    }

    // Search the outer-dim block family: split dim m (dims [0,m) full, dims (m,d)
    // single-index), dim m cut into `nb` contiguous blocks (nb | L[m]) → n_items =
    // nb·∏_{k>m}L[k]. Take the count with the best occupancy fit ≥ team.
    auto const& sh         = f.shape();
    std::size_t best_items = 0;
    std::size_t best_split = 1;
    std::size_t best_block = 1;
    std::size_t above      = 1;  // ∏_{k>m} L[k]
    for (std::size_t m = d - 1; m >= 1; --m) {
        for (std::size_t nb = 1; nb <= sh[m]; ++nb) {
            std::size_t const items = nb * above;
            if (sh[m] % nb == 0 && items >= team && better_slabs(items, best_items, target, team)) {
                best_items = items;
                best_split = m;
                best_block = sh[m] / nb;
            }
        }
        above *= sh[m];
    }
    if (best_items == 0) {  // lattice too small for `team` slabs → finest tiling
        best_items = n / sh[0];
        best_split = 1;
        best_block = 1;
    }
    return {.n_items    = best_items,
            .item_sites = n / best_items,
            .n_sites    = n,
            .split_dim  = best_split,
            .block      = best_block};
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
    int const nthr = request < static_cast<int>(p.n_items) ? request : static_cast<int>(p.n_items);
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
        // NOLINTBEGIN(readability-avoid-nested-conditional-operator) — aligned slab-size chain
        std::size_t const s = p.split_dim == 0    ? (mu == 0 ? p.item_sites : 1)
                              : mu < p.split_dim  ? shape[mu]
                              : mu == p.split_dim ? p.block
                                                  : std::size_t{1};
        // NOLINTEND(readability-avoid-nested-conditional-operator)
        slab += std::to_string(s);
    }
    std::string const line = std::format("{} ({}) → {} ({} sites, {}) slab across {} threads",
                                         full,
                                         p.n_sites,
                                         slab,
                                         p.item_sites,
                                         reticolo::log::human_bytes(p.item_sites * bytes_per_site),
                                         nthr);
    static std::mutex mu_lock;
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
    static std::set<std::string> seen;
    {
        std::scoped_lock const lk(mu_lock);
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
// The `_indexed` form additionally hands the worker the canonical partition
// item index: `worker(item, base, cnt)`. The index is a property of the
// partition, NOT of the executing thread or the schedule — item i covers the
// same site range on every invocation for a fixed (team, slabs_per_thread).
// It exists for state that must be bound one-to-one to a slab (the per-slab
// RNG streams: Hmc's momentum fill draws slab i strictly from site stream i).
//
// `gran` is the kernel's SIMD batch width, kept for documentation and the
// batched kernels' tail handling: the partition itself ignores it (there is ONE
// partition per field, shared by every op), but item sizes are whole products
// of inner dims, so for even lattice extents every item is a whole number of
// any power-of-two batch — batched kernels split vector/scalar tails per item,
// deterministically.
template <class Field, class Worker>
inline void field_visit_indexed(Field const& field, std::size_t /*gran*/, Worker const& worker) {
    Partition const p  = partition(field);
    int const nthreads = traverse_threads(p.n_sites, field.bytes_per_site());
    note_slicing(field.shape().data(), field.ndims(), field.bytes_per_site(), p, nthreads);
    parallel_map(nthreads, p.n_items, [&](std::size_t i) {
        std::size_t const base = i * p.item_sites;
        worker(i, base, std::min(p.item_sites, p.n_sites - base));
    });
}

template <class Field, class Worker>
inline void field_visit(Field const& field, std::size_t gran, Worker const& worker) {
    field_visit_indexed(field, gran, [&](std::size_t /*item*/, std::size_t base, std::size_t cnt) {
        worker(base, cnt);
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
