#pragma once

#include <chrono>
#include <cstddef>

namespace reticolo::bench {

// =============================================================================
//  Adaptive-batch timing: call `body()` repeatedly inside a single timed
//  region, doubling the batch size until total wall time exceeds
//  `k_target_seconds`. Returns the per-call wall time = total / n. The
//  batch approach is essential for sub-microsecond kernels where
//  individual `steady_clock::now()` calls have ~ns overhead and can't
//  resolve a single body() invocation.
//
//  Each measurement is preceded by `k_warmup_calls` warmups to fill
//  caches; warmup time is excluded. To smooth one-off OS jitter, the
//  final timed batch is run `k_trials` times and the *minimum* per-call
//  wall time is returned (best of N — standard microbenchmark practice
//  since slowdowns are noise, never speedups).
// =============================================================================

using bench_clock = std::chrono::steady_clock;

inline double seconds(bench_clock::duration d) noexcept {
    return std::chrono::duration<double>(d).count();
}

// Forces the optimiser to materialise `v` (so the call producing it cannot
// be elided) AND treat all memory as potentially clobbered (so a read-only
// kernel can't be hoisted out of a timing loop because the optimiser
// "proves" the input never changes). Standard microbench trick — see
// e.g. Chandler Carruth's CppCon 2015 talk.
template <class T>
[[gnu::always_inline]] inline void consume(T const& v) noexcept {
    asm volatile("" : : "r,m"(v) : "memory");
}

[[gnu::always_inline]] inline void clobber_memory() noexcept {
    asm volatile("" : : : "memory");
}

constexpr double k_target_seconds = 0.3;
constexpr int k_warmup_calls      = 3;
constexpr int k_min_inner         = 1;
constexpr int k_max_inner         = 1 << 24;  // safety cap
constexpr int k_trials            = 5;

template <class Body>
double time_per_call(Body&& body) {
    for (int i = 0; i < k_warmup_calls; ++i) {
        body();
    }
    // First find a batch size n such that running n calls takes
    // ≥ k_target_seconds.
    int n            = k_min_inner;
    double last_wall = 0.0;
    while (n <= k_max_inner) {
        auto const t0 = bench_clock::now();
        for (int i = 0; i < n; ++i) {
            body();
        }
        last_wall = seconds(bench_clock::now() - t0);
        if (last_wall >= k_target_seconds) {
            break;
        }
        n *= 2;
    }
    // Take the min over k_trials full-batch runs at the calibrated n.
    double best_per_call = last_wall / static_cast<double>(n);
    for (int t = 0; t < k_trials - 1; ++t) {
        auto const t0 = bench_clock::now();
        for (int i = 0; i < n; ++i) {
            body();
        }
        double const per_call =
            seconds(bench_clock::now() - t0) / static_cast<double>(n);
        if (per_call < best_per_call) {
            best_per_call = per_call;
        }
    }
    return best_per_call;
}

}  // namespace reticolo::bench
