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

// Shared hypercubic-lattice geometry: shape, strides, site count, held in a
// process-wide weak-ptr pool so sibling fields reuse one instance. Periodic BCs
// only; `next(s, mu)` / `prev(s, mu)` are COMPUTED from the packed strides (no
// stored neighbour table — the hot kernels roll their own row-nested strided
// offsets, and the tables cost `nsites·d·2·8` bytes for nothing). Supports 1..4
// dimensions. If open BCs are ever needed, they land as a separate
// `OpenIndexing` alongside this class, not as a runtime flag inside it.
class Indexing {
public:
    using SizeVec = std::vector<std::size_t>;

    static std::shared_ptr<Indexing const> acquire(SizeVec shape);

    Indexing(Indexing const&)            = delete;
    Indexing& operator=(Indexing const&) = delete;
    Indexing(Indexing&&)                 = delete;
    Indexing& operator=(Indexing&&)      = delete;
    ~Indexing()                          = default;

    [[nodiscard]] SizeVec const& shape() const noexcept { return shape_; }
    [[nodiscard]] std::size_t ndims() const noexcept { return ndims_; }
    [[nodiscard]] std::size_t nsites() const noexcept { return nsites_; }
    [[nodiscard]] SizeVec const& strides() const noexcept { return strides_; }

    // Periodic +μ̂ / −μ̂ neighbour, COMPUTED from the packed strides — there is no
    // stored neighbour table. `coord[mu] = (s / stride[mu]) % L[mu]` selects the
    // wrap. Cold callers only (observers, tests, D=1); the hot kernels roll their
    // own row-nested strided offsets and never call this per site.
    [[nodiscard]] Site next(Site s, std::size_t mu) const noexcept {
        std::size_t const v = s.value();
        std::size_t const c = (v / strides_[mu]) % shape_[mu];
        return Site{(c + 1 < shape_[mu]) ? v + strides_[mu]
                                         : v - ((shape_[mu] - 1) * strides_[mu])};
    }
    [[nodiscard]] Site prev(Site s, std::size_t mu) const noexcept {
        std::size_t const v = s.value();
        std::size_t const c = (v / strides_[mu]) % shape_[mu];
        return Site{(c > 0) ? v - strides_[mu] : v + ((shape_[mu] - 1) * strides_[mu])};
    }

private:
    explicit Indexing(SizeVec shape);
    void build_();

    SizeVec shape_;
    SizeVec strides_;  // strides_[0]=1, strides_[mu]=∏_{k<mu} L[k]
    std::size_t nsites_ = 0;
    std::size_t ndims_  = 0;
};

namespace impl {

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

}  // namespace impl

inline std::shared_ptr<Indexing const> Indexing::acquire(SizeVec shape) {
    static std::mutex pool_mutex;
    static std::unordered_map<impl::IndexingPoolKey,
                              std::weak_ptr<Indexing const>,
                              impl::IndexingPoolKeyHash>
        pool;

    if (shape.empty()) {
        throw std::invalid_argument{"Indexing: shape must be non-empty"};
    }

    impl::IndexingPoolKey key{std::move(shape)};

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
    // The library supports 1..4 lattice dimensions; the hand-written per-dim
    // strided kernels are sized for that range (gauge additionally needs d≥2).
    if (d > 4) {
        throw std::invalid_argument{"Indexing: ndims must be 1..4"};
    }

    nsites_ = 1;
    for (auto length : shape_) {
        if (length == 0) {
            throw std::invalid_argument{"Indexing: zero-size dimension"};
        }
        nsites_ *= length;
    }

    // Packed strides are the whole geometry now: next()/prev() and every hot
    // kernel derive neighbours from these instead of a stored index table.
    strides_.assign(d, 1);
    for (std::size_t mu = 1; mu < d; ++mu) {
        strides_[mu] = strides_[mu - 1] * shape_[mu - 1];
    }
}

}  // namespace reticolo
