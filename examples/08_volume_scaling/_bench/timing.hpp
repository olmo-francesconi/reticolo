#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <vector>

namespace reticolo::bench {

// Adaptive-batch timing: call `body()` repeatedly inside a single timed
// region, doubling the batch size until total wall time exceeds
// `k_target_seconds`. Returns the per-call wall time = total / n. The
// batch approach is essential for sub-microsecond kernels where
// individual `steady_clock::now()` calls have ~ns overhead and can't
// resolve a single body() invocation.
//
// Each measurement is preceded by `k_warmup_calls` warmups to fill
// caches; warmup time is excluded. To smooth one-off OS jitter, the
// final timed batch is run `k_trials` times and the *minimum* per-call
// wall time is returned (best of N — standard microbenchmark practice
// since slowdowns are noise, never speedups).

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

// Run `body()` until either `budget.max_dofs` total dof updates are
// processed (n_calls × dofs_per_call) or `budget.max_seconds` of wall
// time elapse — whichever comes first. Batch size doubles so clock
// overhead stays negligible even for sub-µs kernels. Returns per-call
// wall time and the total number of calls used; reporting both lets the
// analyser see the sample count behind each timing point.
struct Budget {
    double max_dofs;
    double max_seconds;
};

struct BudgetedResult {
    double wall_s;      // total elapsed / n_calls
    long long n_calls;  // number of body() invocations after warmup
};

template <class Body>
BudgetedResult time_per_call_budgeted(Body&& body, std::size_t dofs_per_call, Budget budget) {
    for (int i = 0; i < k_warmup_calls; ++i) {
        body();
    }
    long long n_calls = 0;
    long long batch   = 1;
    auto const t0     = bench_clock::now();
    double elapsed    = 0.0;
    auto const dofs_d = static_cast<double>(dofs_per_call);
    while (true) {
        for (long long i = 0; i < batch; ++i) {
            body();
        }
        n_calls += batch;
        elapsed = seconds(bench_clock::now() - t0);
        if (static_cast<double>(n_calls) * dofs_d >= budget.max_dofs) {
            break;
        }
        if (elapsed >= budget.max_seconds) {
            break;
        }
        batch = std::min<long long>(batch * 2, k_max_inner);
    }
    return {.wall_s = elapsed / static_cast<double>(n_calls), .n_calls = n_calls};
}

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
        double const per_call = seconds(bench_clock::now() - t0) / static_cast<double>(n);
        if (per_call < best_per_call) {
            best_per_call = per_call;
        }
    }
    return best_per_call;
}

// Distribution of per-call wall time. Unlike `time_per_call` (best-of-N),
// this collects many batch-mean samples and reports mean / 5th / 95th
// percentile, so callers can show the throughput spread rather than just
// the floor. Batch size is calibrated so each batch lasts at least
// `k_sample_batch_s` (≥ 1 call) — that keeps clock overhead negligible
// while still treating each batch as one sample of the distribution. The
// sample count shrinks for expensive kernels so a slow cell (e.g. a large
// gauge force, ~0.4 s/call) stays bounded instead of running for minutes.
struct Stats {
    double mean;  // per-call wall time [s]
    double p05;
    double p95;
};

constexpr double k_sample_batch_s  = 0.005;
constexpr double k_sample_budget_s = 1.5;
constexpr int k_min_samples        = 24;
constexpr int k_max_samples        = 200;

template <class Body>
Stats time_distribution(Body&& body) {
    for (int i = 0; i < k_warmup_calls; ++i) {
        body();
    }
    long long batch   = 1;
    double batch_wall = 0.0;
    while (batch <= k_max_inner) {
        auto const t0 = bench_clock::now();
        for (long long i = 0; i < batch; ++i) {
            body();
        }
        batch_wall = seconds(bench_clock::now() - t0);
        if (batch_wall >= k_sample_batch_s) {
            break;
        }
        batch *= 2;
    }
    int n_samples = static_cast<int>(k_sample_budget_s / std::max(batch_wall, 1e-9));
    n_samples     = std::clamp(n_samples, k_min_samples, k_max_samples);

    std::vector<double> s;
    s.reserve(static_cast<std::size_t>(n_samples));
    for (int i = 0; i < n_samples; ++i) {
        auto const t0 = bench_clock::now();
        for (long long j = 0; j < batch; ++j) {
            body();
        }
        s.push_back(seconds(bench_clock::now() - t0) / static_cast<double>(batch));
    }
    std::sort(s.begin(), s.end());
    double sum = 0.0;
    for (double v : s) {
        sum += v;
    }
    auto const pct = [&](double q) {
        auto const idx = static_cast<std::size_t>(q * static_cast<double>(s.size() - 1));
        return s[idx];
    };
    return {.mean = sum / static_cast<double>(s.size()), .p05 = pct(0.05), .p95 = pct(0.95)};
}

}  // namespace reticolo::bench
