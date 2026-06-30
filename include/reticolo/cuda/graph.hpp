#pragma once

#include <reticolo/cuda/check.hpp>

#include <utility>

#include <cuda_runtime.h>

namespace reticolo::cuda {

// Captures a per-trajectory operation sequence into a CUDA graph once, then
// replays it with a single launch on every subsequent trajectory — paying the
// ~10µs/launch driver overhead at build time instead of per trajectory. The
// trajectory's stream-operation sequence is identical every time (static
// topology, mutable buffer contents), which is exactly what a graph wants.
//
// THE CAPTURE TRAP: capture freezes kernel
// ARGUMENTS by value. Anything that varies per trajectory — the Philox
// trajectory counter above all, also dt if it is ever varied — must be read
// by the kernels from a *device buffer* the host bumps between replays, NOT
// passed as a kernel literal. A baked-in trajectory counter would regenerate
// identical momenta on every replay: a non-ergodic chain that still looks
// thermalized. The `run` body below must honour that.
//
// Non-copyable and non-movable: the instantiated graph bakes raw device
// pointers, so the owning object must not relocate.
class TrajectoryGraph {
public:
    explicit TrajectoryGraph(cudaStream_t stream) : stream_{stream} {}

    TrajectoryGraph(TrajectoryGraph const&)            = delete;
    TrajectoryGraph& operator=(TrajectoryGraph const&) = delete;
    TrajectoryGraph(TrajectoryGraph&&)                 = delete;
    TrajectoryGraph& operator=(TrajectoryGraph&&)      = delete;

    ~TrajectoryGraph() { destroy_(); }

    // Drop the cached executable graph. Call when the trajectory topology
    // changes — i.e. `n_md` changes the node count (re-instantiate, ms-scale).
    // A `tau`-only change needs no invalidation iff the coefficients are fed
    // through a device buffer rather than baked as kernel args.
    void invalidate() noexcept { destroy_(); }

    // Capture `body` into a graph on first call, then replay the cached graph.
    // `body` must enqueue ALL its work on this object's stream and must obey
    // the capture trap above.
    template <class Body>
    void run(Body&& body) {
        if (exec_ == nullptr) {
            capture_(std::forward<Body>(body));
        }
        RETICOLO_CUDA_CHECK(cudaGraphLaunch(exec_, stream_));
    }

    [[nodiscard]] cudaStream_t stream() const noexcept { return stream_; }

private:
    template <class Body>
    void capture_(Body&& body) {
        // Drain any prior eager work on the stream so capture begins cleanly
        // (one-time cost — capture happens once).
        RETICOLO_CUDA_CHECK(cudaStreamSynchronize(stream_));
        RETICOLO_CUDA_CHECK(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
        std::forward<Body>(body)();
        cudaGraph_t graph = nullptr;
        RETICOLO_CUDA_CHECK(cudaStreamEndCapture(stream_, &graph));
        RETICOLO_CUDA_CHECK(cudaGraphInstantiate(&exec_, graph, 0));
        RETICOLO_CUDA_CHECK(cudaGraphDestroy(graph));
    }

    void destroy_() noexcept {
        if (exec_ != nullptr) {
            cudaGraphExecDestroy(exec_);
            exec_ = nullptr;
        }
    }

    cudaStream_t stream_  = nullptr;
    cudaGraphExec_t exec_ = nullptr;
};

}  // namespace reticolo::cuda
