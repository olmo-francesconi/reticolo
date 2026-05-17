#pragma once

#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>

namespace reticolo {

class Site {
public:
    using value_type = std::size_t;

    static constexpr value_type k_invalid_value = std::numeric_limits<value_type>::max();

    constexpr Site() noexcept = default;
    constexpr explicit Site(value_type i) noexcept : i_{i} {}

    [[nodiscard]] constexpr value_type value() const noexcept { return i_; }
    [[nodiscard]] constexpr bool is_valid() const noexcept { return i_ != k_invalid_value; }

    constexpr auto operator<=>(Site const&) const noexcept = default;

    static constexpr Site invalid() noexcept { return Site{k_invalid_value}; }

private:
    value_type i_ = 0;
};

enum class Parity : std::uint8_t {
    Even = 0,
    Odd  = 1,
};

[[nodiscard]] constexpr Parity flip(Parity p) noexcept {
    return p == Parity::Even ? Parity::Odd : Parity::Even;
}

}  // namespace reticolo

template <>
struct std::hash<reticolo::Site> {
    std::size_t operator()(reticolo::Site s) const noexcept {
        return std::hash<reticolo::Site::value_type>{}(s.value());
    }
};
