#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/device_topology.hpp>

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

namespace reticolo::cuda {

// Layout policy: maps a DeviceTopology to the flat element count of the buffer.
// Phase 2 implements ScalarLayout (one element per site, x-fastest, identical
// to Lattice<T>'s AoS order). LinkLayout (Phase 4) and MatrixLayout<G>
// (Phase 5) plug in here without touching DeviceField.
struct ScalarLayout {
    [[nodiscard]] static long flat_count(DeviceTopology const& t) { return t.nsites; }
};

// Gauge link field: ndim·nsites elements, direction-major (link (mu, x) at
// flat index mu·nsites + x) — identical to the host LinkLattice<T> order, so a
// flat copy round-trips exactly via the raw-pointer copy_from_host/to_host.
struct LinkLayout {
    [[nodiscard]] static long flat_count(DeviceTopology const& t) {
        return static_cast<long>(t.ndim) * t.nsites;
    }
};

// Matrix gauge field: ndim·nc·nsites reals, layout [ndim][nc][nsites] with
// nc = G::n_real_components (8 for SU(2), 18 for SU(3)) — identical to the host
// MatrixLinkLattice<G> order, so a flat raw round-trip is exact. Momenta and
// force share this layout (anti-hermitian algebra elements).
template <class G>
struct MatrixLayout {
    [[nodiscard]] static long flat_count(DeviceTopology const& t) {
        return static_cast<long>(t.ndim) * static_cast<long>(G::n_real_components) * t.nsites;
    }
};

// One resident device buffer plus its topology — the device counterpart of
// Lattice<T>. Sibling fields (mom / force / old) are constructed from another
// field's topology so all of them index identically; the topology is a small
// POD held by value (it is also what kernels take by value). Move preserves
// the device pointer (graph-capture invariant — see DeviceBuffer).
template <class T, class Layout = ScalarLayout>
class DeviceField {
public:
    using value_type  = T;
    using layout_type = Layout;

    explicit DeviceField(std::vector<std::size_t> const& shape)
        : topo_{make_device_topology(shape)},
          buf_{static_cast<std::size_t>(Layout::flat_count(topo_))} {}

    // Sibling: same topology, fresh buffer.
    explicit DeviceField(DeviceTopology const& topo)
        : topo_{topo}, buf_{static_cast<std::size_t>(Layout::flat_count(topo_))} {}

    [[nodiscard]] T* data() noexcept { return buf_.data(); }
    [[nodiscard]] T const* data() const noexcept { return buf_.data(); }
    [[nodiscard]] DeviceTopology const& topology() const noexcept { return topo_; }
    [[nodiscard]] long nsites() const noexcept { return topo_.nsites; }
    [[nodiscard]] std::size_t size() const noexcept { return buf_.size(); }

    // Host round-trip. For ScalarLayout the device buffer is `nsites`
    // contiguous elements in the same order as Lattice<T>, so a flat copy is
    // exact; non-scalar layouts will reorder here in later phases.
    void copy_from_host(Lattice<T> const& l, cudaStream_t stream = nullptr) {
        buf_.copy_from_host(l.data(), stream);
    }
    void copy_to_host(Lattice<T>& l, cudaStream_t stream = nullptr) const {
        buf_.copy_to_host(l.data(), stream);
    }

    // Raw host-pointer overloads (caller guarantees `size()` elements) — used to
    // stage momenta drawn by the host RNG before the device RNG lands (Phase 2c).
    void copy_from_host(T const* src, cudaStream_t stream = nullptr) {
        buf_.copy_from_host(src, stream);
    }
    void copy_to_host(T* dst, cudaStream_t stream = nullptr) const {
        buf_.copy_to_host(dst, stream);
    }

private:
    DeviceTopology topo_;
    DeviceBuffer<T> buf_;
};

// flat_size ADL overload — lets the unchanged integrator atoms loop over the
// buffer the same way they do for Lattice<T> / LinkLattice<T>.
template <class T, class Layout>
[[nodiscard]] inline std::size_t flat_size(DeviceField<T, Layout> const& f) noexcept {
    return f.size();
}

}  // namespace reticolo::cuda
