#pragma once

#include <cuda_runtime.h>

namespace reticolo::cuda {

// Deterministic device reductions / vector ops, f64. Fixed launch config so the
// summation order is reproducible run-to-run — required for HMC reversibility
// (a varying reduction tree changes ΔH non-deterministically). Phase 1 uses a
// simple block-partial + fixed-order host finish; Phase 2 fuses s_full+kinetic
// on top of the same primitive.
//
// Defined in src/cuda/reduce.cu. Pointers are device pointers.

// y[i] += a * x[i]
void axpy_f64(double a, double const* x, double* y, long n, cudaStream_t stream = nullptr);

// Σ x[i], accumulated in double with a fixed, reproducible order.
[[nodiscard]] double reduce_sum_f64(double const* x, long n, cudaStream_t stream = nullptr);

// Σ x[i]², same fixed-order determinism. The HMC kinetic term ½Σp² — kept a
// separate primitive (rather than square-into-scratch + reduce_sum) so the
// reduction stays one pass and run-to-run reproducible for reversibility.
[[nodiscard]] double reduce_sumsq_f64(double const* x, long n, cudaStream_t stream = nullptr);

}  // namespace reticolo::cuda
