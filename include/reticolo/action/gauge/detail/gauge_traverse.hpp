#pragma once

#include <reticolo/core/parallel.hpp>

#include <cstddef>
#include <utility>

// Gauge-family traversal shell — the gauge analogue of the site family's
// detail/traversal.hpp. It is the single home for the OpenMP mechanism on gauge
// link fields: the leaf (`Wilson<G>::force_into` / `s_full_uncached`) supplies a
// pure, execution-agnostic per-range worker from the group's formula kernels, and
// these helpers own the `in_traverse_region` region entry + the site-range
// partition. The formula files (action/gauge/formula/*, math/*_ops.hpp) stay free
// of any threading so they remain bit-shareable with the CUDA device path.
//
//   * gauge_traverse_apply  — write-disjoint pass (force, fused kick): each thread
//     runs the worker on one `gran`-aligned contiguous chunk. Because every link
//     is written exactly once and the chunk is batch-aligned, the result is
//     bit-identical to the serial sweep for any thread count.
//   * gauge_traverse_reduce — reduction (Σ Re Tr U_p for s_full): a fixed
//     `gran`-block partition (independent of thread count) fills a partials buffer,
//     summed in canonical block order → thread-count-invariant, like the site
//     reduce_fwd.

namespace reticolo::action::detail {

// Apply `worker(base, cnt)` over [0, nsites) as a write-disjoint pass (force,
// fused kick). Thin gauge-named forwarder to the generic write-disjoint SPMD
// primitive; `gran` is the kernel's site-batch width so batched-vs-tail groupings
// match the serial sweep.
template <class Worker>
inline void gauge_traverse_apply(std::size_t nsites, std::size_t gran, Worker const& worker) {
    reticolo::detail::apply_chunked(nsites, gran, worker);
}

// Reduce `worker(base, cnt) -> double` over [0, nsites) — the Σ Re Tr U_p plane
// sum for s_full. Thin gauge-named forwarder to the generic deterministic
// blocked reduction (fixed gran-blocks → thread-count-invariant).
template <class Worker>
[[nodiscard]] inline double
gauge_traverse_reduce(std::size_t nsites, std::size_t gran, Worker const& worker) {
    return reticolo::detail::reduce_blocks(nsites, gran, worker);
}

}  // namespace reticolo::action::detail
