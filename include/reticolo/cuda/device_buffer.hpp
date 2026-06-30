#pragma once

#include <reticolo/cuda/check.hpp>

#include <cstddef>

#include <cuda_runtime.h>

namespace reticolo::cuda {

// Minimal owning device buffer — a run-lifetime `cudaMalloc` region with RAII
// teardown. `DeviceField<T, Layout>` builds on this.
//
// Move-only, and move *preserves the device pointer value* (it transfers, not
// reallocates). This is load-bearing for graph capture: a captured graph bakes
// the raw `d_` pointer into its kernel-argument nodes, so a buffer that moved
// (e.g. when its owner moved) must keep the same address. Never reallocate a
// buffer whose pointer is baked into a live graphExec.
template <class T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t n) : n_{n} {
        if (n_ != 0) {
            RETICOLO_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_), n_ * sizeof(T)));
        }
    }

    DeviceBuffer(DeviceBuffer const&)            = delete;
    DeviceBuffer& operator=(DeviceBuffer const&) = delete;

    DeviceBuffer(DeviceBuffer&& o) noexcept : d_{o.d_}, n_{o.n_} {
        o.d_ = nullptr;
        o.n_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            reset_();
            d_   = o.d_;
            n_   = o.n_;
            o.d_ = nullptr;
            o.n_ = 0;
        }
        return *this;
    }

    ~DeviceBuffer() { reset_(); }

    [[nodiscard]] T* data() noexcept { return d_; }
    [[nodiscard]] T const* data() const noexcept { return d_; }
    [[nodiscard]] std::size_t size() const noexcept { return n_; }

    void copy_from_host(T const* src, cudaStream_t stream = nullptr) {
        RETICOLO_CUDA_CHECK(
            cudaMemcpyAsync(d_, src, n_ * sizeof(T), cudaMemcpyHostToDevice, stream));
    }
    void copy_to_host(T* dst, cudaStream_t stream = nullptr) const {
        RETICOLO_CUDA_CHECK(
            cudaMemcpyAsync(dst, d_, n_ * sizeof(T), cudaMemcpyDeviceToHost, stream));
    }

private:
    void reset_() noexcept {
        if (d_ != nullptr) {
            cudaFree(d_);  // teardown path: implicit-sync free, errors are unrecoverable here
            d_ = nullptr;
        }
        n_ = 0;
    }

    T* d_          = nullptr;
    std::size_t n_ = 0;
};

}  // namespace reticolo::cuda
