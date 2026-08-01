#pragma once

namespace reticolo::obs {

// The two kernel shapes the observable engine fuses — one concept per entry
// point, each used as a constraint (obs::reduce / obs::reduce_nn and their
// device twins cuda::reduce / cuda::reduce_nn). Result types are deliberately
// unconstrained: each lane accumulates in whatever its kernel returns (double,
// cplx, …).

// Per-site: callable on a field value. The unit `reduce` folds.
template <class K, class T>
concept SiteKernel = requires(K const& k, T self) { k(self); };

// Neighbour-aware: callable on (self, agg) where `agg` is the Policy-selected
// neighbour aggregate. The unit `reduce_nn` folds — the same shape an action
// leaf's action_kernel/force_kernel consumes.
template <class K, class T>
concept NnKernel = requires(K const& k, T self, T agg) { k(self, agg); };

}  // namespace reticolo::obs
