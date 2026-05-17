#include <reticolo/core/bc.hpp>

#include <catch2/catch_test_macros.hpp>

using reticolo::Bc;
using reticolo::BcMask;

TEST_CASE("BcMask defaults to all-periodic for given ndims", "[bc]") {
    BcMask const m{4};
    REQUIRE(m.ndims() == 4);
    REQUIRE(m.all_periodic());
    for (std::size_t mu = 0; mu < 4; ++mu) {
        REQUIRE(m[mu] == Bc::Periodic);
        REQUIRE_FALSE(m.affects_topology(mu));
    }
}

TEST_CASE("BcMask from initializer_list captures per-direction BCs", "[bc]") {
    BcMask const m{Bc::Periodic, Bc::Open, Bc::Antiperiodic};
    REQUIRE(m.ndims() == 3);
    REQUIRE_FALSE(m.all_periodic());
    REQUIRE(m[0] == Bc::Periodic);
    REQUIRE(m[1] == Bc::Open);
    REQUIRE(m[2] == Bc::Antiperiodic);

    REQUIRE_FALSE(m.affects_topology(0));
    REQUIRE(m.affects_topology(1));
    REQUIRE_FALSE(m.affects_topology(2));
}

TEST_CASE("BcMask equality compares structure", "[bc]") {
    REQUIRE(BcMask(3, Bc::Periodic) == BcMask{Bc::Periodic, Bc::Periodic, Bc::Periodic});
    REQUIRE_FALSE(BcMask(3, Bc::Periodic) == BcMask(3, Bc::Open));
    REQUIRE_FALSE(BcMask{Bc::Periodic, Bc::Open} == BcMask{Bc::Open, Bc::Periodic});
}
