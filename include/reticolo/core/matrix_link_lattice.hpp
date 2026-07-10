#pragma once

#include <reticolo/core/indexing.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/core/parallel.hpp>
#include <reticolo/core/site.hpp>

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

namespace reticolo {

// Per-component storage span (in elements) for a matrix-link field of `nsites`
// links and scalar size `elem`. The site axis of each SoA component slab is
// padded from `nsites` up to `link_span >= nsites` so that consecutive
// components (which sit `link_span·elem` bytes apart) do NOT collapse onto one
// cache set. A Wilson SU(3) link is 18 components at stride link_span; with the
// natural packed stride `link_span = nsites` and a power-of-two lattice,
// `nsites·8` is a clean multiple of the L1 way-size, so all 18 components alias
// into a single set and an 8-way cache cannot hold one link — the origin of the
// 16⁴/32⁴ throughput dips. The fix is target-agnostic: round the span up to a
// whole cache line (so component starts stay line-aligned) and then force an
// ODD number of lines per component. An odd line-count makes the component
// stride coprime to any power-of-two set count, so the 18 component starts fan
// out across distinct sets instead of colliding. The pad is a few lines per
// component (~0.01% of the buffer); the holes [nsites, link_span) are never
// read or written by any kernel — every access is blk[k·link_span + s] with
// s < nsites — so the field's physics is byte-identical to the packed layout.
[[nodiscard]] inline std::size_t padded_link_span(std::size_t nsites, std::size_t elem) noexcept {
    constexpr std::size_t line_bytes = exec::k_cache_line_bytes;
    std::size_t const line_elems     = elem != 0 ? (line_bytes / elem) : 1;
    if (line_elems <= 1) {
        return nsites;
    }
    std::size_t span = ((nsites + line_elems - 1) / line_elems) * line_elems;
    if (((span / line_elems) & 1U) == 0) {  // even #lines/component → still aliases
        span += line_elems;                 // bump to an odd line count → set spread
    }
    return span;
}

// Opt-in switch for the site-axis cache padding (see padded_link_span). OFF by
// default, so every gauge lattice is PACKED (link_span == nsites) and its buffer
// is byte-for-byte the historic layout — existing tests, config IO, and the
// packed `_slab` helpers need no change. A production HMC app flips this ON once
// at startup, BEFORE constructing any gauge field; the field and the HMC
// momentum/force/rollback siblings it spawns (all built from the same shape while
// the flag is on) derive the same deterministic link_span, so their component
// strides match. It changes only the in-memory storage stride — physics is
// identical, and config IO stays packed regardless — so it is safe to leave a
// run's result untouched while making the hot loop cache-friendly. Set once,
// single-threaded at startup; not meant to be toggled mid-run.
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
inline bool g_gauge_link_padding = false;

// The component stride a fresh gauge lattice of `nsites` links (scalar size
// `elem`) should use: padded when the opt-in flag is set, else exactly nsites.
[[nodiscard]] inline std::size_t gauge_link_span(std::size_t nsites, std::size_t elem) noexcept {
    return g_gauge_link_padding ? padded_link_span(nsites, elem) : nsites;
}

// Matrix-valued link field on a hypercubic (periodic) lattice. Each link
// carries one N×N complex group element (SU(N), or any GaugeGroup model);
// storage is split into `n_real_components = 2*N*N` real arrays per
// direction, each contiguous over a PADDED site axis of `link_span >= nsites`
// elements (see padded_link_span). The flat buffer layout is
//
//      [ndim][n_real_components][link_span]   (lex order)
//      flat = ((mu * nc) + k) * link_span + site_index(x)
//
// i.e. the existing direction-major LinkLattice pattern, refined one level
// further into per-component SoA slabs. For a fixed direction mu each per-
// component array (`mu_block_data(mu)` + k·link_span) is a plain stride-1
// buffer — the Wilson plaquette plane loops can walk all 2*N*N component streams in
// lockstep, giving the compiler clean stride-1 FMA chains across sites
// without any intrinsics. The site-axis padding keeps those component streams
// off a shared cache set (see padded_link_span); it is invisible to the
// physics because every kernel indexes site s ∈ [0, nsites) at stride
// link_span, never touching the [nsites, link_span) tail.
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
          link_span_{gauge_link_span(idx_->nsites(), sizeof(T))},
          data_(idx_->ndims() * n_real_components * link_span_, T{}) {
        require_gauge_dims_(idx_->ndims());
        log::info("init",
                  "MatrixLinkLattice<{}, {}>  shape={}  links={}",
                  G::name,
                  scalar_name<T>(),
                  shape_str(idx_->shape()),
                  idx_->ndims() * idx_->nsites());
    }

    explicit MatrixLinkLattice(std::shared_ptr<Indexing const> idx)
        : idx_{std::move(idx)}, link_span_{gauge_link_span(idx_->nsites(), sizeof(T))},
          data_(idx_->ndims() * n_real_components * link_span_, T{}) {
        require_gauge_dims_(idx_->ndims());
    }

    MatrixLinkLattice(MatrixLinkLattice const&)                = default;
    MatrixLinkLattice(MatrixLinkLattice&&) noexcept            = default;
    MatrixLinkLattice& operator=(MatrixLinkLattice const&)     = default;
    MatrixLinkLattice& operator=(MatrixLinkLattice&&) noexcept = default;
    ~MatrixLinkLattice()                                       = default;

    // Component-slab stride (elements between component k and k+1 of one link,
    // and between direction blocks). Padded ≥ nsites so components don't share a
    // cache set (see padded_link_span). This is the multiplier every gauge
    // kernel must use for the `k·span + s` addressing — NOT nsites, which is the
    // valid-site count. The two coincide only when no padding is applied.
    [[nodiscard]] std::size_t link_span() const noexcept { return link_span_; }

    // Pointer to the full nc-component block for direction mu (length
    // nc·link_span). Convenient for passing all component pointers to a group
    // slab kernel by base + stride (the stride being link_span()).
    [[nodiscard]] T* mu_block_data(std::size_t mu) noexcept {
        return data_.data() + (mu * n_real_components * link_span_);
    }
    [[nodiscard]] T const* mu_block_data(std::size_t mu) const noexcept {
        return data_.data() + (mu * n_real_components * link_span_);
    }

    [[nodiscard]] Site next(Site s, std::size_t mu) const noexcept { return idx_->next(s, mu); }
    [[nodiscard]] Site prev(Site s, std::size_t mu) const noexcept { return idx_->prev(s, mu); }

    [[nodiscard]] SizeVec const& shape() const noexcept { return idx_->shape(); }
    [[nodiscard]] std::size_t ndims() const noexcept { return idx_->ndims(); }
    [[nodiscard]] std::size_t nsites() const noexcept { return idx_->nsites(); }

    // Per-site storage footprint (bytes) = ndims · components · sizeof(T).
    // Reported by the lattice so slab logging reflects the real (large) gauge
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
    // A gauge link field lives in 2 ≤ d ≤ 4 dimensions: at least one μ<ν plaquette
    // plane, and the strided Wilson kernels are sized for up to four directions.
    static void require_gauge_dims_(std::size_t ndims) {
        if (ndims < 2 || ndims > 4) {
            throw std::invalid_argument{
                "MatrixLinkLattice: gauge link field requires 2 <= ndims <= 4"};
        }
    }

    std::shared_ptr<Indexing const> idx_;
    std::size_t link_span_;
    std::vector<T> data_;
};

}  // namespace reticolo
