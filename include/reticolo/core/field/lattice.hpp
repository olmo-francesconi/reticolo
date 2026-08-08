#pragma once

#include <reticolo/core/field/lattice_geometry.hpp>
#include <reticolo/core/field/site.hpp>
#include <reticolo/core/log/log.hpp>

#include <cstddef>
#include <memory>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace reticolo {

// Lattice<T> — owning value-semantic field container on a periodic hypercubic
// lattice. One value of type `T` per site, stored in a flat std::vector.
//
//     Lattice<double> a{{16, 16, 16}};
//     Lattice<double> b = a;       // deep copy: data AND geometry
//     b[Site{0}] = 42;             // modifies only b
//
// The geometry (shape, packed strides, site count) is inlined into the field
// through `impl::LatticeGeometry` — it is not a separate object, and there is
// no pool. A sibling buffer is built from the same shape, which costs four
// multiplies:
//
//     Lattice<double> mom{phi.shape()};   // HMC's mom / force / old_field
template <class T>
class Lattice : public impl::LatticeGeometry {
public:
    using value_type = T;
    using SizeVec    = std::vector<std::size_t>;

    // Default-fill a fresh lattice on a new shape (always periodic).
    explicit Lattice(SizeVec const& shape)
        : impl::LatticeGeometry{std::span<std::size_t const>{shape}}, data_(nsites(), T{}) {
        log_construct_();
    }

    Lattice(SizeVec const& shape, T fill)
        : impl::LatticeGeometry{std::span<std::size_t const>{shape}},
          data_(nsites(), std::move(fill)) {
        log_construct_();
    }

    // Sibling-lattice constructors: same shape, fresh storage (the HMC pattern —
    // mom, force and old_field are built from the field's own shape()).
    explicit Lattice(std::span<std::size_t const> shape)
        : impl::LatticeGeometry{shape}, data_(nsites(), T{}) {
        log_construct_();
    }

    Lattice(std::span<std::size_t const> shape, T fill)
        : impl::LatticeGeometry{shape}, data_(nsites(), std::move(fill)) {
        log_construct_();
    }

    // Value semantics: deep-copy data and geometry (the geometry is 80 bytes inline).
    Lattice(Lattice const&)                = default;
    Lattice(Lattice&&) noexcept            = default;
    Lattice& operator=(Lattice const&)     = default;
    Lattice& operator=(Lattice&&) noexcept = default;
    ~Lattice()                             = default;

    // A scalar field has one value per site, so the element accessor is
    // `operator[](Site)`. The link field's accessor is `operator()(Site, mu)`
    // — the differing arity (one DOF per site vs one per site-direction) is
    // why the two field types use different operators by design.
    [[nodiscard]] T& operator[](Site s) noexcept { return data_[s.value()]; }
    [[nodiscard]] T const& operator[](Site s) const noexcept { return data_[s.value()]; }

    // shape() / ndims() / nsites() / next() / prev() come from the geometry base.

    // Per-site storage footprint (bytes) — the threading threshold / chunk size are
    // derived from this, so a memory-bound decision reflects the real footprint.
    [[nodiscard]] static constexpr std::size_t bytes_per_site() noexcept { return sizeof(T); }

    [[nodiscard]] auto sites() const noexcept {
        return std::views::iota(std::size_t{0}, nsites()) |
               std::views::transform([](std::size_t i) { return Site{i}; });
    }

    [[nodiscard]] T* data() noexcept { return data_.data(); }
    [[nodiscard]] T const* data() const noexcept { return data_.data(); }

    [[nodiscard]] auto begin() noexcept { return data_.begin(); }
    [[nodiscard]] auto end() noexcept { return data_.end(); }
    [[nodiscard]] auto begin() const noexcept { return data_.begin(); }
    [[nodiscard]] auto end() const noexcept { return data_.end(); }

private:
    void log_construct_() const {
        log::info("init",
                  "Lattice<{}>  shape={}  sites={}",
                  scalar_name<T>(),
                  shape_str(shape()),
                  nsites());
    }

    std::vector<T> data_;
};

}  // namespace reticolo
