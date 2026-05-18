#pragma once

#include <reticolo/core/indexing.hpp>
#include <reticolo/core/site.hpp>

#include <cstddef>
#include <memory>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace reticolo {

template <class T>
class Lattice {
public:
    using value_type = T;
    using SizeVec    = Indexing::SizeVec;

    // Default-fill a fresh lattice on a new shape (always periodic).
    explicit Lattice(SizeVec shape)
        : idx_{Indexing::acquire(std::move(shape))}, data_(idx_->nsites(), T{}) {}

    Lattice(SizeVec shape, T fill)
        : idx_{Indexing::acquire(std::move(shape))},
          data_(idx_->nsites(), std::move(fill)) {}

    // Sibling-lattice constructors: reuse an existing Indexing
    // (the HMC pattern — mom, force, old_field share the field's topology).
    explicit Lattice(std::shared_ptr<Indexing const> idx)
        : idx_{std::move(idx)}, data_(idx_->nsites(), T{}) {}

    Lattice(std::shared_ptr<Indexing const> idx, T fill)
        : idx_{std::move(idx)}, data_(idx_->nsites(), std::move(fill)) {}

    // Value semantics: deep-copy data, share Indexing.
    Lattice(Lattice const&)                = default;
    Lattice(Lattice&&) noexcept            = default;
    Lattice& operator=(Lattice const&)     = default;
    Lattice& operator=(Lattice&&) noexcept = default;
    ~Lattice()                             = default;

    [[nodiscard]] T&       operator[](Site s) noexcept { return data_[s.value()]; }
    [[nodiscard]] T const& operator[](Site s) const noexcept { return data_[s.value()]; }

    [[nodiscard]] Site   next(Site s, std::size_t mu) const noexcept { return idx_->next(s, mu); }
    [[nodiscard]] Site   prev(Site s, std::size_t mu) const noexcept { return idx_->prev(s, mu); }
    [[nodiscard]] Parity parity_of(Site s) const noexcept { return idx_->parity_of(s); }

    [[nodiscard]] SizeVec const& shape() const noexcept { return idx_->shape(); }
    [[nodiscard]] std::size_t    ndims() const noexcept { return idx_->ndims(); }
    [[nodiscard]] std::size_t    nsites() const noexcept { return idx_->nsites(); }

    [[nodiscard]] auto sites() const noexcept {
        return std::views::iota(std::size_t{0}, nsites()) |
               std::views::transform([](std::size_t i) { return Site{i}; });
    }
    [[nodiscard]] std::span<Site const> even_sites() const noexcept { return idx_->even_sites(); }
    [[nodiscard]] std::span<Site const> odd_sites() const noexcept { return idx_->odd_sites(); }

    [[nodiscard]] T*       data() noexcept { return data_.data(); }
    [[nodiscard]] T const* data() const noexcept { return data_.data(); }

    [[nodiscard]] auto begin() noexcept { return data_.begin(); }
    [[nodiscard]] auto end() noexcept { return data_.end(); }
    [[nodiscard]] auto begin() const noexcept { return data_.begin(); }
    [[nodiscard]] auto end() const noexcept { return data_.end(); }

    [[nodiscard]] std::shared_ptr<Indexing const> indexing() const noexcept { return idx_; }

    // Non-owning accessor for hot kernels — bypasses the shared_ptr copy
    // (refcount atomics) that calling `indexing()` would incur. Returned
    // reference is valid for the life of this `Lattice`.
    [[nodiscard]] Indexing const& indexing_ref() const noexcept { return *idx_; }

private:
    std::shared_ptr<Indexing const> idx_;
    std::vector<T>                  data_;
};

}  // namespace reticolo
