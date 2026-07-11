#pragma once

#include <algorithm>

#ifdef _OPENMP
    #include <omp.h>
#endif

namespace reticolo::orch {

// How the ambient OpenMP thread budget is split across a worker ensemble.
//
// There is no total-threads flag: the budget is read from the environment
// (OMP_NUM_THREADS / omp_set_num_threads) and split into an outer worker team
// (`concurrency` workers running at once) and inner per-worker teams (`m`
// threads each, e.g. an HMC's site-level traverse). `nested` records whether
// the second active OpenMP level must be enabled (only when m > 1).
//
// Worker-level parallelism is ~linear and bandwidth-efficient; site-level
// threading inside a worker is sublinear and BW-capped, so the default policy
// saturates workers first and spills leftover threads into inner teams only
// when workers run out — never past the inner-scaling knee `m_max`.
struct ThreadPlan {
    int concurrency = 0;      // workers run at once (outer omp team); 0 = ambient
    int m           = 1;      // threads per worker (inner team)
    bool nested     = false;  // enable a 2nd active OpenMP level (m > 1)
};

// Allocate the ambient thread budget over `n_workers`.
//   requested_m = 1  → one thread per worker; whole budget goes to concurrency
//   requested_m = 0  → auto-balance: m = clamp(T / n_workers, 1, m_max)
//   requested_m > 1  → pin a larger inner team of that size
// concurrency is filled from the remaining budget; m·concurrency ≤ T.
[[nodiscard]] inline ThreadPlan plan_threads(int n_workers, int requested_m = 0, int m_max = 4) {
    int total = 0;
#ifdef _OPENMP
    total = omp_get_max_threads();
#endif
    if (total <= 0) {
        total = 1;
    }
    int m = 1;
    if (requested_m > 0) {
        m = requested_m;
    } else if (requested_m == 0) {
        m = std::clamp(total / std::max(n_workers, 1), 1, m_max);
    }
    int const conc = std::clamp(total / std::max(m, 1), 1, std::max(n_workers, 1));
    return {.concurrency = conc, .m = m, .nested = m > 1};
}

}  // namespace reticolo::orch
