#pragma once

#include <reticolo/orch/concepts.hpp>
#include <reticolo/orch/thread_plan.hpp>

#include <cstddef>
#include <memory>
#include <vector>

#ifdef _OPENMP
    #include <omp.h>
#endif

namespace reticolo::orch {

// The one concurrent primitive every orchestration shares: run `body(i, w)`
// over every worker in `workers` at once, under `plan`. `body` receives the
// worker's index `i` (so a caller can key per-worker scratch / output buffers)
// and a reference to the worker `w` itself.
//
// This is worker-level (outer) parallelism: the loop runs on `plan.concurrency`
// lanes and, when `plan.nested`, a second OpenMP active level is enabled so a
// worker's own inner team (m > 1) can spawn without oversubscribing. Each
// worker's sampler pins its own inner thread count independently (e.g. an HMC
// binds its team via exec::team_scope), so the two levels compose.
//
// Deliberately concurrent-only: it does NOT bundle a serial "drain" step. The
// thread-unsafe output pass (HDF5 writes, logging) stays a plain serial loop in
// the caller, right next to the schedule — the orchestration owns its loop, the
// spine only owns the fan-out. Without OpenMP the body simply runs serially.
template <class Worker, class Body>
    requires orch::Worker<Worker>
void parallel_workers(std::vector<std::unique_ptr<Worker>>& workers,
                      ThreadPlan const& plan,
                      Body&& body) {
    std::size_t const n = workers.size();
#ifdef _OPENMP
    if (plan.nested) {
        omp_set_max_active_levels(2);
    }
    int const outer_nt = plan.concurrency > 0 ? plan.concurrency : omp_get_max_threads();
    #pragma omp parallel for schedule(dynamic, 1) num_threads(outer_nt)
    for (std::size_t i = 0; i < n; ++i) {
        body(i, *workers[i]);
    }
#else
    (void)plan;
    for (std::size_t i = 0; i < n; ++i) {
        body(i, *workers[i]);
    }
#endif
}

}  // namespace reticolo::orch
