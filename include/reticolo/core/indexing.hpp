#pragma once

#include <reticolo/core/bc.hpp>
#include <reticolo/core/site.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace reticolo {

class Indexing {
public:
    using SizeVec = std::vector<std::size_t>;
    using SiteVec = std::vector<Site>;

    // Acquire or construct the shared Indexing for (shape, bcs).
    // Instances are pool-shared via weak_ptr; identical (shape, bcs)
    // requests return the same shared_ptr.
    static std::shared_ptr<Indexing const> acquire(SizeVec shape, BcMask bcs = {});

    Indexing(Indexing const&)            = delete;
    Indexing& operator=(Indexing const&) = delete;
    Indexing(Indexing&&)                 = delete;
    Indexing& operator=(Indexing&&)      = delete;
    ~Indexing()                          = default;

    [[nodiscard]] SizeVec const& shape() const noexcept { return shape_; }
    [[nodiscard]] BcMask const& bcs() const noexcept { return bcs_; }
    [[nodiscard]] std::size_t ndims() const noexcept { return shape_.size(); }
    [[nodiscard]] std::size_t nsites() const noexcept { return nsites_; }
    [[nodiscard]] bool all_periodic() const noexcept { return all_periodic_; }

    [[nodiscard]] Site next(Site s, std::size_t mu) const noexcept {
        return Site{next_[s.value() * ndims() + mu]};
    }
    [[nodiscard]] Site prev(Site s, std::size_t mu) const noexcept {
        return Site{prev_[s.value() * ndims() + mu]};
    }
    [[nodiscard]] Parity parity_of(Site s) const noexcept {
        return parity_[s.value()] == 0 ? Parity::Even : Parity::Odd;
    }
    [[nodiscard]] bool is_interior(Site s) const noexcept {
        return all_periodic_ || skin_lookup_[s.value()] == 0;
    }

    [[nodiscard]] SiteVec const& bulk_sites() const noexcept { return bulk_; }
    [[nodiscard]] SiteVec const& skin_sites() const noexcept { return skin_; }
    [[nodiscard]] SiteVec const& even_sites() const noexcept { return even_; }
    [[nodiscard]] SiteVec const& odd_sites() const noexcept { return odd_; }

private:
    Indexing(SizeVec shape, BcMask bcs);
    void build_();

    SizeVec shape_;
    BcMask bcs_;
    bool all_periodic_  = true;
    std::size_t nsites_ = 0;

    std::vector<Site::value_type> next_;
    std::vector<Site::value_type> prev_;
    std::vector<std::uint8_t> parity_;
    std::vector<std::uint8_t> skin_lookup_;

    SiteVec bulk_;
    SiteVec skin_;
    SiteVec even_;
    SiteVec odd_;
};

namespace detail {

struct IndexingPoolKey {
    Indexing::SizeVec shape;
    BcMask bcs;
    bool operator==(IndexingPoolKey const&) const = default;
};

struct IndexingPoolKeyHash {
    std::size_t operator()(IndexingPoolKey const& k) const noexcept {
        std::size_t h = 0;
        auto mix = [&h](std::size_t v) { h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6U) + (h >> 2U); };
        for (auto s : k.shape) {
            mix(std::hash<std::size_t>{}(s));
        }
        for (Bc b : k.bcs.as_vector()) {
            mix(std::hash<std::uint8_t>{}(static_cast<std::uint8_t>(b)));
        }
        return h;
    }
};

}  // namespace detail

inline std::shared_ptr<Indexing const> Indexing::acquire(SizeVec shape, BcMask bcs) {
    static std::mutex pool_mutex;
    static std::unordered_map<detail::IndexingPoolKey,
                              std::weak_ptr<Indexing const>,
                              detail::IndexingPoolKeyHash>
        pool;

    if (shape.empty()) {
        throw std::invalid_argument{"Indexing: shape must be non-empty"};
    }
    if (bcs.ndims() == 0) {
        bcs = BcMask(shape.size(), Bc::Periodic);
    }
    if (bcs.ndims() != shape.size()) {
        throw std::invalid_argument{"Indexing: BcMask dims must match shape"};
    }

    detail::IndexingPoolKey key{std::move(shape), std::move(bcs)};

    std::lock_guard const lock{pool_mutex};
    if (auto it = pool.find(key); it != pool.end()) {
        if (auto sp = it->second.lock()) {
            return sp;
        }
        pool.erase(it);
    }
    auto sp = std::shared_ptr<Indexing const>(new Indexing(key.shape, key.bcs));
    pool.emplace(std::move(key), sp);
    return sp;
}

inline Indexing::Indexing(SizeVec shape, BcMask bcs)
    : shape_{std::move(shape)}, bcs_{std::move(bcs)} {
    build_();
}

inline void Indexing::build_() {
    std::size_t const d = shape_.size();

    nsites_ = 1;
    for (auto length : shape_) {
        if (length == 0) {
            throw std::invalid_argument{"Indexing: zero-size dimension"};
        }
        nsites_ *= length;
    }
    all_periodic_ = bcs_.all_periodic();

    next_.assign(nsites_ * d, Site::k_invalid_value);
    prev_.assign(nsites_ * d, Site::k_invalid_value);
    parity_.assign(nsites_, std::uint8_t{0});
    skin_lookup_.assign(nsites_, std::uint8_t{0});

    std::vector<std::size_t> coord(d, 0);
    std::vector<std::size_t> stride(d, 1);
    for (std::size_t mu = 1; mu < d; ++mu) {
        stride[mu] = stride[mu - 1] * shape_[mu - 1];
    }

    for (std::size_t s = 0; s < nsites_; ++s) {
        std::size_t r          = s;
        std::size_t parity_sum = 0;
        bool is_skin           = false;
        for (std::size_t mu = 0; mu < d; ++mu) {
            coord[mu] = r % shape_[mu];
            r /= shape_[mu];
            parity_sum += coord[mu];
            if (!all_periodic_ && bcs_.affects_topology(mu) &&
                (coord[mu] == 0 || coord[mu] + 1 == shape_[mu])) {
                is_skin = true;
            }
        }
        parity_[s]      = static_cast<std::uint8_t>(parity_sum & 1U);
        skin_lookup_[s] = is_skin ? std::uint8_t{1} : std::uint8_t{0};

        for (std::size_t mu = 0; mu < d; ++mu) {
            std::size_t const length = shape_[mu];

            if (coord[mu] + 1 < length) {
                next_[s * d + mu] = s + stride[mu];
            } else if (bcs_[mu] == Bc::Open) {
                next_[s * d + mu] = Site::k_invalid_value;
            } else {
                next_[s * d + mu] = s + stride[mu] - length * stride[mu];
            }

            if (coord[mu] > 0) {
                prev_[s * d + mu] = s - stride[mu];
            } else if (bcs_[mu] == Bc::Open) {
                prev_[s * d + mu] = Site::k_invalid_value;
            } else {
                prev_[s * d + mu] = s + (length - 1) * stride[mu];
            }
        }
    }

    bulk_.clear();
    skin_.clear();
    even_.clear();
    odd_.clear();
    bulk_.reserve(nsites_);
    even_.reserve(nsites_ / 2 + 1);
    odd_.reserve(nsites_ / 2 + 1);
    for (std::size_t s = 0; s < nsites_; ++s) {
        Site const site{s};
        if (skin_lookup_[s] != 0) {
            skin_.push_back(site);
        } else {
            bulk_.push_back(site);
        }
        if (parity_[s] == 0) {
            even_.push_back(site);
        } else {
            odd_.push_back(site);
        }
    }
}

}  // namespace reticolo
