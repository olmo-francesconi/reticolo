#pragma once

#include <reticolo/core/field/site.hpp>

#include <array>
#include <complex>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace reticolo {

// Render a SizeVec (or any range of integral) as "L0×L1×L2…" for log lines.
// Uses Unicode × (multiplication sign), not 'x'.
template <class R>
[[nodiscard]] inline std::string shape_str(R const& shape) {
    std::string out;
    for (std::size_t i = 0; auto v : shape) {
        if (i++ != 0) {
            out.append("×");
        }
        out.append(std::to_string(v));
    }
    return out;
}

// Compact human-readable name for the scalar value type used by an action /
// field. Used in log lines so apps don't need to spell it out by hand.
// (Defined here rather than in field_traits.hpp so headers that *describe*
// themselves — Lattice, MatrixLinkLattice, … — can use it without pulling in
// field_traits, which would create a circular include.)
template <class T>
[[nodiscard]] consteval std::string_view scalar_name() noexcept {
    if constexpr (std::is_same_v<T, double>) {
        return "double";
    } else if constexpr (std::is_same_v<T, float>) {
        return "float";
    } else if constexpr (std::is_same_v<T, std::complex<double>>) {
        return "cdouble";
    } else if constexpr (std::is_same_v<T, std::complex<float>>) {
        return "cfloat";
    } else {
        return "?";
    }
}

namespace impl {

// Hypercubic lattice geometry — shape, packed strides, site count — owned BY the
// field rather than held in a separate object the field points at.
//
// This was a pooled `Indexing` shared through a `shared_ptr`, and that paid for
// itself while it carried neighbour TABLES (`nsites·d·2·8` bytes — tens of MB at
// production volumes). Those are gone: `next`/`prev` are closed-form on the
// strides, and the hot nests roll their own row-nested offsets. What was left
// was 64 bytes reached through a process-wide mutex, an atomic refcount and a
// pointer chase, saving 240 bytes across HMC's four buffers against 40 MB of
// payload each. Building this from a shape is four multiplies; sharing it saved
// nothing, so there is nothing to share and no lifetime to couple.
//
// A private BASE rather than a member, so the accessors are the field's own API
// (`l.nsites()`, `l.shape()`) and no geometry object appears in any signature.
// Sibling fields are now built from a shape: `Lattice<T> mom{phi.shape()}`.
//
// Periodic BCs only, 1..4 dimensions. If open BCs are ever needed they land as a
// separate geometry base, not as a runtime flag in this one.
class LatticeGeometry {
public:
    static constexpr std::size_t k_max_dims = 4;

    // A view, not a container: the extents live inline. Build a shape with
    // `std::vector<std::size_t>` (the field ctors take one); read one back
    // through this.
    [[nodiscard]] std::span<std::size_t const> shape() const noexcept {
        return {shape_.data(), ndims_};
    }
    [[nodiscard]] std::size_t ndims() const noexcept { return ndims_; }
    [[nodiscard]] std::size_t nsites() const noexcept { return nsites_; }

    // Periodic +μ̂ / −μ̂ neighbour, computed from the packed strides — there is no
    // stored neighbour table. `coord[mu] = (s / stride[mu]) % L[mu]` selects the
    // wrap. Cold callers only (observers, tests, D=1); the hot kernels roll their
    // own row-nested strided offsets and never call this per site.
    [[nodiscard]] Site next(Site s, std::size_t mu) const noexcept {
        std::size_t const v = s.value();
        std::size_t const c = (v / stride_[mu]) % shape_[mu];
        return Site{(c + 1 < shape_[mu]) ? v + stride_[mu] : v - ((shape_[mu] - 1) * stride_[mu])};
    }
    [[nodiscard]] Site prev(Site s, std::size_t mu) const noexcept {
        std::size_t const v = s.value();
        std::size_t const c = (v / stride_[mu]) % shape_[mu];
        return Site{(c > 0) ? v - stride_[mu] : v + ((shape_[mu] - 1) * stride_[mu])};
    }

    // Two fields are geometrically interchangeable iff their extents match; this
    // replaces the old "do they point at the same pooled Indexing" test, which
    // was only ever a proxy for it.
    [[nodiscard]] bool same_geometry(LatticeGeometry const& o) const noexcept {
        return ndims_ == o.ndims_ && nsites_ == o.nsites_ && shape_ == o.shape_;
    }

protected:
    LatticeGeometry() = default;
    explicit LatticeGeometry(std::span<std::size_t const> shape) { build_(shape); }

private:
    void build_(std::span<std::size_t const> shape) {
        if (shape.empty()) {
            throw std::invalid_argument{"Lattice: shape must be non-empty"};
        }
        // The library supports 1..4 lattice dimensions; the hand-written per-dim
        // strided kernels are sized for that range (gauge additionally needs d≥2).
        if (shape.size() > k_max_dims) {
            throw std::invalid_argument{"Lattice: ndims must be 1..4"};
        }
        ndims_             = shape.size();
        std::size_t packed = 1;
        for (std::size_t mu = 0; mu < ndims_; ++mu) {
            if (shape[mu] == 0) {
                throw std::invalid_argument{"Lattice: every extent must be non-zero"};
            }
            shape_[mu]  = shape[mu];
            stride_[mu] = packed;  // stride_[0]=1, stride_[mu]=∏_{k<mu} L[k]
            packed *= shape[mu];
        }
        nsites_ = packed;
    }

    std::array<std::size_t, k_max_dims> shape_{};
    std::array<std::size_t, k_max_dims> stride_{};
    std::size_t nsites_ = 0;
    std::size_t ndims_  = 0;
};

}  // namespace impl
}  // namespace reticolo
