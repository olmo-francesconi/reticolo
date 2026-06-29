#pragma once

#include <reticolo/core/indexing.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/site.hpp>

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
// Value semantics with one twist:
//
//     Lattice<double> a{{16, 16, 16}};
//     Lattice<double> b = a;       // ← deep-copies the FIELD DATA
//                                  //   but SHARES the Indexing neighbour table
//     b[Site{0}] = 42;             //   modifies only b's data
//
// The Indexing (neighbour pointers, parity labels) is immutable and pooled by
// shape, so sharing it costs only a shared_ptr increment. Two lattices of the
// same shape automatically share — `Lattice<T> mom{phi.indexing()}` (the
// "sibling" ctor) constructs a fresh field without rebuilding the topology.
// HMC uses this for its mom/force/old_field buffers: one neighbour table,
// four lattices.
//
// If you ever need a fully independent copy whose Indexing is its own object
// (rare — you'd be paying for nothing), construct from a fresh shape:
//     Lattice<double> b{a.shape()};
//     std::ranges::copy(a, b.begin());
template <class T>
class Lattice {
public:
    using value_type = T;
    using SizeVec    = Indexing::SizeVec;

    // Default-fill a fresh lattice on a new shape (always periodic).
    explicit Lattice(SizeVec shape)
        : idx_{Indexing::acquire(std::move(shape))}, data_(idx_->nsites(), T{}) {
        log_construct_();
    }

    Lattice(SizeVec shape, T fill)
        : idx_{Indexing::acquire(std::move(shape))}, data_(idx_->nsites(), std::move(fill)) {
        log_construct_();
    }

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

    // A scalar field has one value per site, so the element accessor is
    // `operator[](Site)`. The link field's accessor is `operator()(Site, mu)`
    // — the differing arity (one DOF per site vs one per site-direction) is
    // why the two field types use different operators by design.
    [[nodiscard]] T& operator[](Site s) noexcept { return data_[s.value()]; }
    [[nodiscard]] T const& operator[](Site s) const noexcept { return data_[s.value()]; }

    [[nodiscard]] Site next(Site s, std::size_t mu) const noexcept { return idx_->next(s, mu); }
    [[nodiscard]] Site prev(Site s, std::size_t mu) const noexcept { return idx_->prev(s, mu); }
    [[nodiscard]] Parity parity_of(Site s) const noexcept { return idx_->parity_of(s); }

    [[nodiscard]] SizeVec const& shape() const noexcept { return idx_->shape(); }
    [[nodiscard]] std::size_t ndims() const noexcept { return idx_->ndims(); }
    [[nodiscard]] std::size_t nsites() const noexcept { return idx_->nsites(); }

    [[nodiscard]] auto sites() const noexcept {
        return std::views::iota(std::size_t{0}, nsites()) |
               std::views::transform([](std::size_t i) { return Site{i}; });
    }

    // Visit every elementary update slot — one DOF per site for a scalar
    // field. Body is called as `body(T& ref, Site x)`. ref-first is required
    // so the twin call on LinkLattice can append a `mu` arg captured by a
    // trailing parameter pack in the body (`auto... loc` after `T& ref`).
    template <class Body>
    void for_each_update(Body&& body) {
        T* const d          = data_.data();
        std::size_t const n = idx_->nsites();
        for (std::size_t i = 0; i < n; ++i) {
            body(d[i], Site{i});
        }
    }
    [[nodiscard]] std::span<Site const> even_sites() const noexcept { return idx_->even_sites(); }
    [[nodiscard]] std::span<Site const> odd_sites() const noexcept { return idx_->odd_sites(); }

    [[nodiscard]] T* data() noexcept { return data_.data(); }
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
    void log_construct_() const {
        log::info("init",
                  "Lattice<{}>  shape={}  sites={}",
                  scalar_name<T>(),
                  shape_str(idx_->shape()),
                  idx_->nsites());
    }

    std::shared_ptr<Indexing const> idx_;
    std::vector<T> data_;
};

}  // namespace reticolo
