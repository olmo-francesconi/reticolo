#pragma once

#include <reticolo/core/indexing.hpp>
#include <reticolo/core/site.hpp>

#include <cstddef>
#include <memory>
#include <ranges>
#include <utility>
#include <vector>

namespace reticolo {

// =============================================================================
//  Link field on a hypercubic (periodic) lattice. Each site x owns `ndim`
//  outgoing links theta_mu(x), mu = 0..ndim-1, where theta_mu(x) is the
//  oriented link x -> x + mu_hat. Every link in the lattice appears exactly
//  once.
//
//  Storage is a single flat std::vector<T> of size `ndim * nsites`, indexed
//  site-major:
//        flat = ndim * site_index(x) + mu
//  so a site's ndim outgoing links are contiguous. The shape-pooled
//  Indexing is shared with the sibling site `Lattice<F>` of the same shape
//  (e.g. field, momentum, force in HMC share one neighbour table).
// =============================================================================

template <class T>
class LinkLattice {
public:
    using value_type = T;
    using SizeVec    = Indexing::SizeVec;

    explicit LinkLattice(SizeVec shape)
        : idx_{Indexing::acquire(std::move(shape))}, data_(idx_->ndims() * idx_->nsites(), T{}) {}

    LinkLattice(SizeVec shape, T fill)
        : idx_{Indexing::acquire(std::move(shape))},
          data_(idx_->ndims() * idx_->nsites(), std::move(fill)) {}

    explicit LinkLattice(std::shared_ptr<Indexing const> idx)
        : idx_{std::move(idx)}, data_(idx_->ndims() * idx_->nsites(), T{}) {}

    LinkLattice(std::shared_ptr<Indexing const> idx, T fill)
        : idx_{std::move(idx)}, data_(idx_->ndims() * idx_->nsites(), std::move(fill)) {}

    LinkLattice(LinkLattice const&)                = default;
    LinkLattice(LinkLattice&&) noexcept            = default;
    LinkLattice& operator=(LinkLattice const&)     = default;
    LinkLattice& operator=(LinkLattice&&) noexcept = default;
    ~LinkLattice()                                 = default;

    [[nodiscard]] T& operator()(Site s, std::size_t mu) noexcept {
        return data_[(idx_->ndims() * s.value()) + mu];
    }
    [[nodiscard]] T const& operator()(Site s, std::size_t mu) const noexcept {
        return data_[(idx_->ndims() * s.value()) + mu];
    }

    [[nodiscard]] Site next(Site s, std::size_t mu) const noexcept { return idx_->next(s, mu); }
    [[nodiscard]] Site prev(Site s, std::size_t mu) const noexcept { return idx_->prev(s, mu); }

    [[nodiscard]] SizeVec const& shape() const noexcept { return idx_->shape(); }
    [[nodiscard]] std::size_t ndims() const noexcept { return idx_->ndims(); }
    [[nodiscard]] std::size_t nsites() const noexcept { return idx_->nsites(); }
    [[nodiscard]] std::size_t nlinks() const noexcept { return data_.size(); }

    [[nodiscard]] T* data() noexcept { return data_.data(); }
    [[nodiscard]] T const* data() const noexcept { return data_.data(); }

    [[nodiscard]] auto begin() noexcept { return data_.begin(); }
    [[nodiscard]] auto end() noexcept { return data_.end(); }
    [[nodiscard]] auto begin() const noexcept { return data_.begin(); }
    [[nodiscard]] auto end() const noexcept { return data_.end(); }

    [[nodiscard]] auto sites() const noexcept {
        return std::views::iota(std::size_t{0}, nsites()) |
               std::views::transform([](std::size_t i) { return Site{i}; });
    }

    [[nodiscard]] std::shared_ptr<Indexing const> indexing() const noexcept { return idx_; }
    [[nodiscard]] Indexing const& indexing_ref() const noexcept { return *idx_; }

private:
    std::shared_ptr<Indexing const> idx_;
    std::vector<T> data_;
};

}  // namespace reticolo
