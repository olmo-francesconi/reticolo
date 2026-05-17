#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <utility>
#include <vector>

namespace reticolo {

enum class Bc : std::uint8_t {
    Periodic,
    Antiperiodic,
    Open,
};

class BcMask {
public:
    BcMask() = default;

    explicit BcMask(std::size_t ndims, Bc default_bc = Bc::Periodic) : bcs_(ndims, default_bc) {}

    BcMask(std::initializer_list<Bc> bcs) : bcs_(bcs) {}

    explicit BcMask(std::vector<Bc> bcs) : bcs_(std::move(bcs)) {}

    [[nodiscard]] std::size_t ndims() const noexcept { return bcs_.size(); }

    [[nodiscard]] Bc operator[](std::size_t mu) const noexcept { return bcs_[mu]; }

    [[nodiscard]] bool all_periodic() const noexcept {
        return std::ranges::all_of(bcs_, [](Bc b) { return b == Bc::Periodic; });
    }

    [[nodiscard]] bool affects_topology(std::size_t mu) const noexcept {
        return bcs_[mu] == Bc::Open;
    }

    [[nodiscard]] std::vector<Bc> const& as_vector() const noexcept { return bcs_; }

    bool operator==(BcMask const&) const = default;

private:
    std::vector<Bc> bcs_;
};

}  // namespace reticolo
