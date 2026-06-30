#pragma once

#include <cuda_runtime.h>

// Thread-local "current stream" for the device backend. The integrator atoms
// (cuda::drift_field / kick_add) and DeviceAction launch on it, so cuda::Hmc
// can route an entire MD trajectory onto one capture stream without threading a
// stream argument through the unchanged integrators. Default is the null
// stream, so eager (non-captured) behaviour is exactly as before.
//
// CUDA graph capture requires every operation to be enqueued on the (single,
// non-default) stream being captured; this is the mechanism that puts them
// there.

namespace reticolo::cuda {

inline cudaStream_t& current_stream_ref() {
    static thread_local cudaStream_t s = nullptr;
    return s;
}

[[nodiscard]] inline cudaStream_t current_stream() {
    return current_stream_ref();
}

inline void set_current_stream(cudaStream_t s) {
    current_stream_ref() = s;
}

// RAII: set the current stream for a scope, restore the previous on exit.
class ScopedStream {
public:
    explicit ScopedStream(cudaStream_t s) : prev_{current_stream()} { set_current_stream(s); }
    ~ScopedStream() { set_current_stream(prev_); }
    ScopedStream(ScopedStream const&)            = delete;
    ScopedStream& operator=(ScopedStream const&) = delete;
    ScopedStream(ScopedStream&&)                 = delete;
    ScopedStream& operator=(ScopedStream&&)      = delete;

private:
    cudaStream_t prev_;
};

}  // namespace reticolo::cuda
