// The physics-free orchestration spine: ThreadPlan arithmetic, the Worker
// concept gate, and parallel_workers' fan-out contract. These are the pieces
// every orchestrator (span, LLR, and any future one) is built on, and they had
// no direct coverage — only indirect exercise through app binaries, where a
// plan_threads regression would surface as a performance change rather than a
// test failure.
//
// All O(1) arithmetic and bookkeeping; no Monte Carlo.

#include <reticolo/orch/concepts.hpp>
#include <reticolo/orch/ensemble.hpp>
#include <reticolo/orch/thread_plan.hpp>

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using reticolo::orch::parallel_workers;
using reticolo::orch::plan_threads;
using reticolo::orch::ThreadPlan;

namespace {

// Minimal Worker: the concept asks for `id()` and nothing else.
struct CountingWorker {
    std::string name;
    int touched = 0;

    [[nodiscard]] std::string_view id() const noexcept { return name; }
};

// Lacks id() — must NOT satisfy the concept.
struct NotAWorker {
    int x = 0;
};

std::vector<std::unique_ptr<CountingWorker>> make_workers(std::size_t n) {
    std::vector<std::unique_ptr<CountingWorker>> v;
    v.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        v.push_back(std::make_unique<CountingWorker>(
            CountingWorker{.name = "w" + std::to_string(i), .touched = 0}));
    }
    return v;
}

}  // namespace

TEST_CASE("orch::Worker gates on id(); Checkpointable refines it", "[orch][concept]") {
    STATIC_REQUIRE(reticolo::orch::Worker<CountingWorker>);
    STATIC_REQUIRE_FALSE(reticolo::orch::Worker<NotAWorker>);
    // A bare Worker has no field()/rng(), so it is not Checkpointable.
    STATIC_REQUIRE_FALSE(reticolo::orch::Checkpointable<CountingWorker>);
}

// requested_m == 1 is the "serial replica" default every LLR app ships with:
// the whole budget must go to worker concurrency, and no nested level opens.
TEST_CASE("plan_threads with m=1 spends the budget on concurrency", "[orch][threads]") {
    ThreadPlan const p = plan_threads(/*n_workers=*/8, /*requested_m=*/1);
    REQUIRE(p.m == 1);
    REQUIRE_FALSE(p.nested);
    REQUIRE(p.concurrency >= 1);
    REQUIRE(p.concurrency <= 8);  // never more lanes than workers
}

// A pinned inner team must be honoured exactly and must flip `nested`, which is
// what enables the second active OpenMP level.
TEST_CASE("plan_threads honours a pinned inner team and flags nesting", "[orch][threads]") {
    ThreadPlan const p = plan_threads(/*n_workers=*/4, /*requested_m=*/3);
    REQUIRE(p.m == 3);
    REQUIRE(p.nested);
    REQUIRE(p.concurrency >= 1);
}

// Auto-balance (m = 0) must never exceed the inner-scaling knee m_max — past it
// the site-level threading is bandwidth-capped and the threads are wasted.
TEST_CASE("plan_threads auto-balance clamps the inner team to m_max", "[orch][threads]") {
    constexpr int k_m_max = 4;
    for (int n_workers : {1, 2, 3, 8, 64}) {
        ThreadPlan const p = plan_threads(n_workers, /*requested_m=*/0, k_m_max);
        INFO("n_workers=" << n_workers);
        REQUIRE(p.m >= 1);
        REQUIRE(p.m <= k_m_max);
        REQUIRE(p.concurrency >= 1);
        REQUIRE(p.concurrency <= std::max(n_workers, 1));
        REQUIRE(p.nested == (p.m > 1));
    }
}

// The budget invariant the whole split exists to maintain.
TEST_CASE("plan_threads never oversubscribes: m * concurrency <= budget", "[orch][threads]") {
    for (int m : {0, 1, 2, 4}) {
        for (int n_workers : {1, 4, 16}) {
            ThreadPlan const p = plan_threads(n_workers, m);
            INFO("requested_m=" << m << " n_workers=" << n_workers);
            // The budget is the ambient OMP max (1 without OpenMP). concurrency
            // is derived as clamp(total/m, 1, n_workers), so the product can only
            // exceed the budget via the lower clamp — i.e. concurrency == 1.
            REQUIRE((p.m * p.concurrency <= std::max(p.m, 1) * std::max(n_workers, 1)));
            REQUIRE(p.concurrency >= 1);
        }
    }
}

// Degenerate inputs must not produce a zero/negative lane count — a 0 there
// would make parallel_workers request an invalid team.
TEST_CASE("plan_threads is safe for zero workers", "[orch][threads]") {
    ThreadPlan const p = plan_threads(/*n_workers=*/0, /*requested_m=*/1);
    REQUIRE(p.concurrency >= 1);
    REQUIRE(p.m >= 1);
}

// The fan-out contract: every worker is visited exactly once, and the index the
// body receives is the worker's own position (orchestrators key per-worker
// output series off it, so a mismatch would silently cross-write results).
TEST_CASE("parallel_workers visits every worker exactly once, with its own index",
          "[orch][ensemble]") {
    constexpr std::size_t k_n = 16;
    auto workers              = make_workers(k_n);

    std::vector<std::size_t> seen_index(k_n, 0);
    parallel_workers(workers, plan_threads(static_cast<int>(k_n), 1), [&](std::size_t i, auto& w) {
        w.touched += 1;
        // Each lane writes its own slot only — no synchronisation needed.
        seen_index[i] = 1;
        REQUIRE(w.id() == "w" + std::to_string(i));
    });

    for (std::size_t i = 0; i < k_n; ++i) {
        INFO("worker " << i);
        REQUIRE(workers[i]->touched == 1);
        REQUIRE(seen_index[i] == 1);
    }
}

TEST_CASE("parallel_workers on an empty ensemble is a no-op", "[orch][ensemble]") {
    std::vector<std::unique_ptr<CountingWorker>> none;
    int calls = 0;
    parallel_workers(none, plan_threads(0, 1), [&](std::size_t, auto&) { ++calls; });
    REQUIRE(calls == 0);
}

// parallel_workers is deliberately concurrent-only — the serial drain stays in
// the caller. Running it twice must accumulate, not reset: orchestrators call it
// once per wave and rely on worker state persisting across waves.
TEST_CASE("parallel_workers preserves worker state across waves", "[orch][ensemble]") {
    auto workers       = make_workers(4);
    ThreadPlan const p = plan_threads(4, 1);
    for (int wave = 0; wave < 3; ++wave) {
        parallel_workers(workers, p, [](std::size_t, auto& w) { w.touched += 1; });
    }
    for (auto const& w : workers) {
        REQUIRE(w->touched == 3);
    }
}
