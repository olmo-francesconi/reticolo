#pragma once

#include <reticolo/cuda/check.hpp>

#include <cstddef>

#include <cuda_runtime.h>

namespace reticolo::cuda {

// Page-locked (pinned) host buffer. The per-trajectory ΔH copy-back and the
// measurement transfers go through pinned memory so cudaMemcpy doesn't stage
// through a pageable bounce buffer on the critical path. RAII; move-only.
template <class T>
class PinnedBuffer {
public:
    PinnedBuffer() = default;

    explicit PinnedBuffer(std::size_t n) : n_{n} {
        if (n_ != 0) {
            RETICOLO_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&p_), n_ * sizeof(T)));
        }
    }

    PinnedBuffer(PinnedBuffer const&)            = delete;
    PinnedBuffer& operator=(PinnedBuffer const&) = delete;

    PinnedBuffer(PinnedBuffer&& o) noexcept : p_{o.p_}, n_{o.n_} {
        o.p_ = nullptr;
        o.n_ = 0;
    }
    PinnedBuffer& operator=(PinnedBuffer&& o) noexcept {
        if (this != &o) {
            reset_();
            p_   = o.p_;
            n_   = o.n_;
            o.p_ = nullptr;
            o.n_ = 0;
        }
        return *this;
    }

    ~PinnedBuffer() { reset_(); }

    [[nodiscard]] T* data() noexcept { return p_; }
    [[nodiscard]] T const* data() const noexcept { return p_; }
    [[nodiscard]] std::size_t size() const noexcept { return n_; }

private:
    void reset_() noexcept {
        if (p_ != nullptr) {
            cudaFreeHost(p_);
            p_ = nullptr;
        }
        n_ = 0;
    }

    T* p_          = nullptr;
    std::size_t n_ = 0;
};

}  // namespace reticolo::cuda
