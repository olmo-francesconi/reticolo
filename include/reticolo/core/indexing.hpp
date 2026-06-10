#pragma once

#include <reticolo/core/site.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>
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
// themselves — Lattice, LinkLattice, … — can use it without pulling in
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

// Periodic-only neighbour table for a hypercubic lattice. The library does
// not currently support open / antiperiodic boundaries — every `next(s, mu)`
// and `prev(s, mu)` wraps around. Hot kernels therefore never need a
// validity check in the inner loop. If open BCs are ever needed, they land
// as a separate `OpenIndexing` (or a constexpr policy template parameter)
// alongside this class, not as a runtime flag inside it.
class Indexing {
public:
    using SizeVec = std::vector<std::size_t>;
    using SiteVec = std::vector<Site>;

    static std::shared_ptr<Indexing const> acquire(SizeVec shape);

    Indexing(Indexing const&)            = delete;
    Indexing& operator=(Indexing const&) = delete;
    Indexing(Indexing&&)                 = delete;
    Indexing& operator=(Indexing&&)      = delete;
    ~Indexing()                          = default;

    [[nodiscard]] SizeVec const& shape() const noexcept { return shape_; }
    // Plain member load, not shape_.size(): next()/prev() multiply by ndims()
    // on the address path of every neighbour lookup, and the vector-size
    // indirection (two loads + sub + shift) would sit on that critical path.
    [[nodiscard]] std::size_t ndims() const noexcept { return ndims_; }
    [[nodiscard]] std::size_t nsites() const noexcept { return nsites_; }

    [[nodiscard]] Site next(Site s, std::size_t mu) const noexcept {
        return Site{next_[(s.value() * ndims()) + mu]};
    }
    [[nodiscard]] Site prev(Site s, std::size_t mu) const noexcept {
        return Site{prev_[(s.value() * ndims()) + mu]};
    }
    [[nodiscard]] Parity parity_of(Site s) const noexcept {
        return parity_[s.value()] == 0 ? Parity::Even : Parity::Odd;
    }

    [[nodiscard]] SiteVec const& even_sites() const noexcept { return even_; }
    [[nodiscard]] SiteVec const& odd_sites() const noexcept { return odd_; }

    // Raw neighbour-table pointers for hot kernels. Layout: `[site * ndims + mu]`.
    // Stable for the life of this `Indexing` (immutable post-construction).
    [[nodiscard]] Site::value_type const* next_data() const noexcept { return next_.data(); }
    [[nodiscard]] Site::value_type const* prev_data() const noexcept { return prev_.data(); }

private:
    explicit Indexing(SizeVec shape);
    void build_();

    SizeVec shape_;
    std::size_t nsites_ = 0;

    std::vector<Site::value_type> next_;
    std::vector<Site::value_type> prev_;
    std::vector<std::uint8_t> parity_;

    SiteVec even_;
    SiteVec odd_;

    // Declared last so the hot members above (next_/prev_ headers) keep
    // their cache-line placement; see ndims() for why this is a member.
    std::size_t ndims_ = 0;
};

namespace detail {

struct IndexingPoolKey {
    Indexing::SizeVec shape;
    bool operator==(IndexingPoolKey const&) const = default;
};

struct IndexingPoolKeyHash {
    std::size_t operator()(IndexingPoolKey const& k) const noexcept {
        std::size_t h = 0;
        auto mix = [&h](std::size_t v) { h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U); };
        for (auto s : k.shape) {
            mix(std::hash<std::size_t>{}(s));
        }
        return h;
    }
};

}  // namespace detail

inline std::shared_ptr<Indexing const> Indexing::acquire(SizeVec shape) {
    static std::mutex pool_mutex;
    static std::unordered_map<detail::IndexingPoolKey,
                              std::weak_ptr<Indexing const>,
                              detail::IndexingPoolKeyHash>
        pool;

    if (shape.empty()) {
        throw std::invalid_argument{"Indexing: shape must be non-empty"};
    }

    detail::IndexingPoolKey key{std::move(shape)};

    std::lock_guard const lock{pool_mutex};
    if (auto it = pool.find(key); it != pool.end()) {
        if (auto sp = it->second.lock()) {
            return sp;
        }
        pool.erase(it);
    }
    auto sp = std::shared_ptr<Indexing const>(new Indexing(key.shape));
    pool.emplace(std::move(key), sp);
    return sp;
}

inline Indexing::Indexing(SizeVec shape) : shape_{std::move(shape)} {
    build_();
}

inline void Indexing::build_() {
    std::size_t const d = shape_.size();
    ndims_              = d;

    nsites_ = 1;
    for (auto length : shape_) {
        if (length == 0) {
            throw std::invalid_argument{"Indexing: zero-size dimension"};
        }
        nsites_ *= length;
    }

    next_.assign(nsites_ * d, 0);
    prev_.assign(nsites_ * d, 0);
    parity_.assign(nsites_, std::uint8_t{0});

    std::vector<std::size_t> coord(d, 0);
    std::vector<std::size_t> stride(d, 1);
    for (std::size_t mu = 1; mu < d; ++mu) {
        stride[mu] = stride[mu - 1] * shape_[mu - 1];
    }

    for (std::size_t s = 0; s < nsites_; ++s) {
        std::size_t r          = s;
        std::size_t parity_sum = 0;
        for (std::size_t mu = 0; mu < d; ++mu) {
            coord[mu] = r % shape_[mu];
            r /= shape_[mu];
            parity_sum += coord[mu];
        }
        parity_[s] = static_cast<std::uint8_t>(parity_sum & 1U);

        for (std::size_t mu = 0; mu < d; ++mu) {
            std::size_t const length = shape_[mu];
            next_[(s * d) + mu] =
                (coord[mu] + 1 < length) ? s + stride[mu] : s + stride[mu] - (length * stride[mu]);
            prev_[(s * d) + mu] =
                (coord[mu] > 0) ? s - stride[mu] : s + ((length - 1) * stride[mu]);
        }
    }

    even_.clear();
    odd_.clear();
    even_.reserve(nsites_ / 2 + 1);
    odd_.reserve(nsites_ / 2 + 1);
    for (std::size_t s = 0; s < nsites_; ++s) {
        Site const site{s};
        if (parity_[s] == 0) {
            even_.push_back(site);
        } else {
            odd_.push_back(site);
        }
    }
}

}  // namespace reticolo
