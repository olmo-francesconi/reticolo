#pragma once

#include <reticolo/cuda/macros.hpp>

#include <cstddef>
#include <vector>

namespace reticolo::cuda {

inline constexpr int kMaxDim = 6;

// Periodic hypercubic indexing, computed closed-form per thread rather than
// gathered from a neighbour table — passed BY VALUE into kernels (lands in the
// constant param bank, broadcasts to the warp). Mirrors the CPU `Indexing`
// stride convention exactly: stride[0] = 1 (x fastest), flat index = Σ c_mu·stride_mu.
//
// next/prev are RETICOLO_HD so they run on the device in kernels and on the
// host in tests (where they can be checked against the reference `Indexing`).
struct DeviceTopology {
    int ndim             = 0;
    long nsites          = 0;
    long shape[kMaxDim]  = {};
    long stride[kMaxDim] = {};

    [[nodiscard]] RETICOLO_HD long next(long s, int mu) const {
        long const c = (s / stride[mu]) % shape[mu];
        return (c + 1 < shape[mu]) ? (s + stride[mu]) : (s - ((shape[mu] - 1) * stride[mu]));
    }
    [[nodiscard]] RETICOLO_HD long prev(long s, int mu) const {
        long const c = (s / stride[mu]) % shape[mu];
        return (c > 0) ? (s - stride[mu]) : (s + ((shape[mu] - 1) * stride[mu]));
    }
};

// Host-side builder from a shape (matches reticolo::Indexing's column-major /
// x-fastest layout). Host-only; not called from device code.
[[nodiscard]] inline DeviceTopology make_device_topology(std::vector<std::size_t> const& shape) {
    DeviceTopology t;
    t.ndim  = static_cast<int>(shape.size());
    long st = 1;
    long ns = 1;
    for (int mu = 0; mu < t.ndim; ++mu) {
        t.shape[mu]  = static_cast<long>(shape[static_cast<std::size_t>(mu)]);
        t.stride[mu] = st;
        st *= t.shape[mu];
        ns *= t.shape[mu];
    }
    t.nsites = ns;
    return t;
}

}  // namespace reticolo::cuda
