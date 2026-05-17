#include <reticolo/core/site.hpp>

#include <functional>
#include <unordered_set>

#include <catch2/catch_test_macros.hpp>

using reticolo::flip;
using reticolo::Parity;
using reticolo::Site;

TEST_CASE("Site default-constructs to zero and is valid", "[site]") {
    Site const s{};
    REQUIRE(s.value() == 0);
    REQUIRE(s.is_valid());
}

TEST_CASE("Site::invalid() is sentinel and compares ordered", "[site]") {
    Site const a{3};
    Site const b{7};
    Site const inv = Site::invalid();

    REQUIRE_FALSE(inv.is_valid());
    REQUIRE(a < b);
    REQUIRE(a != b);
    REQUIRE(a == Site{3});
}

TEST_CASE("Site is hashable and usable in unordered_set", "[site]") {
    std::unordered_set<Site> seen;
    for (std::size_t i = 0; i < 10; ++i) {
        seen.insert(Site{i});
    }
    REQUIRE(seen.size() == 10);
    REQUIRE(seen.contains(Site{4}));
    REQUIRE_FALSE(seen.contains(Site{42}));
}

TEST_CASE("flip(Parity) swaps Even <-> Odd", "[site]") {
    REQUIRE(flip(Parity::Even) == Parity::Odd);
    REQUIRE(flip(Parity::Odd) == Parity::Even);
}
