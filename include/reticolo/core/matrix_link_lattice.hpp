#pragma once

#include <reticolo/core/indexing.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/site.hpp>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace reticolo {

// Matrix-valued link field on a hypercubic (periodic) lattice. Each link
// carries one N×N complex group element (SU(N), or any GaugeGroup model);
// storage is split into `n_real_components = 2*N*N` real arrays per
// direction, each contiguous over the nsites site axis. The flat buffer
// layout is
//
//      [ndim][n_real_components][nsites]      (lex order)
//      flat = ((mu * nc) + k) * nsites + site_index(x)
//
// i.e. the existing direction-major LinkLattice pattern, refined one level
// further into per-component SoA slabs. For a fixed direction mu the per-
// component array `mu_comp_data(mu, k)` is a plain stride-1 buffer — the
// Wilson plaquette plane loops can walk all 2*N*N component streams in
// lockstep, giving the compiler clean stride-1 FMA chains across sites
// without any intrinsics.
//
// The momentum field for matrix-group HMC has the same shape (algebra
// elements stored as anti-hermitian traceless N×N matrices), so the same
// type doubles as the momentum buffer; the Group concept's `kinetic_slab`
// / `sample_algebra_slab` are what interpret the components.

template <class G, class T = double>
class MatrixLinkLattice {
public:
    using value_type = T;
    using group_type = G;
    using SizeVec    = Indexing::SizeVec;

    static constexpr std::size_t n_real_components = G::n_real_components;

    explicit MatrixLinkLattice(SizeVec shape)
        : idx_{Indexing::acquire(std::move(shape))},
          data_(idx_->ndims() * n_real_components * idx_->nsites(), T{}) {
        log::info("init",
                  "MatrixLinkLattice<{}, {}>  shape={}  links={}",
                  G::name,
                  scalar_name<T>(),
                  shape_str(idx_->shape()),
                  idx_->ndims() * idx_->nsites());
    }

    explicit MatrixLinkLattice(std::shared_ptr<Indexing const> idx)
        : idx_{std::move(idx)}, data_(idx_->ndims() * n_real_components * idx_->nsites(), T{}) {}

    MatrixLinkLattice(MatrixLinkLattice const&)                = default;
    MatrixLinkLattice(MatrixLinkLattice&&) noexcept            = default;
    MatrixLinkLattice& operator=(MatrixLinkLattice const&)     = default;
    MatrixLinkLattice& operator=(MatrixLinkLattice&&) noexcept = default;
    ~MatrixLinkLattice()                                       = default;

    // Pointer to the contiguous nsites array holding the k-th real component
    // of all links in direction mu. Stride-1, length nsites.
    [[nodiscard]] T* mu_comp_data(std::size_t mu, std::size_t k) noexcept {
        return data_.data() + ((mu * n_real_components + k) * idx_->nsites());
    }
    [[nodiscard]] T const* mu_comp_data(std::size_t mu, std::size_t k) const noexcept {
        return data_.data() + ((mu * n_real_components + k) * idx_->nsites());
    }

    // Pointer to the full nc-component block for direction mu (length nc*nsites).
    // Convenient for passing all component pointers to a group slab kernel by
    // base + stride.
    [[nodiscard]] T* mu_block_data(std::size_t mu) noexcept {
        return data_.data() + (mu * n_real_components * idx_->nsites());
    }
    [[nodiscard]] T const* mu_block_data(std::size_t mu) const noexcept {
        return data_.data() + (mu * n_real_components * idx_->nsites());
    }

    [[nodiscard]] Site next(Site s, std::size_t mu) const noexcept { return idx_->next(s, mu); }
    [[nodiscard]] Site prev(Site s, std::size_t mu) const noexcept { return idx_->prev(s, mu); }

    [[nodiscard]] SizeVec const& shape() const noexcept { return idx_->shape(); }
    [[nodiscard]] std::size_t ndims() const noexcept { return idx_->ndims(); }
    [[nodiscard]] std::size_t nsites() const noexcept { return idx_->nsites(); }

    // Per-site storage footprint (bytes) = ndims · components · sizeof(T). Drives
    // the threading threshold / chunk size so they reflect the real (large) gauge
    // footprint rather than a bare site count.
    [[nodiscard]] std::size_t bytes_per_site() const noexcept {
        return idx_->ndims() * n_real_components * sizeof(T);
    }

    // Number of group elements (one per (site, mu)). Distinct from the raw
    // component count returned by flat_size() / data range.
    [[nodiscard]] std::size_t nlinks() const noexcept { return idx_->ndims() * idx_->nsites(); }

    // Number of real doubles in the underlying buffer. Used by snapshot/
    // rollback and any raw-buffer fast paths.
    [[nodiscard]] std::size_t ncomponents() const noexcept { return data_.size(); }

    [[nodiscard]] T* data() noexcept { return data_.data(); }
    [[nodiscard]] T const* data() const noexcept { return data_.data(); }

    [[nodiscard]] auto begin() noexcept { return data_.begin(); }
    [[nodiscard]] auto end() noexcept { return data_.end(); }
    [[nodiscard]] auto begin() const noexcept { return data_.begin(); }
    [[nodiscard]] auto end() const noexcept { return data_.end(); }

    [[nodiscard]] std::shared_ptr<Indexing const> indexing() const noexcept { return idx_; }
    [[nodiscard]] Indexing const& indexing_ref() const noexcept { return *idx_; }

private:
    std::shared_ptr<Indexing const> idx_;
    std::vector<T> data_;
};

}  // namespace reticolo
