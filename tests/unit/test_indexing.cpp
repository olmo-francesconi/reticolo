#include <reticolo/core/indexing.hpp>
#include <reticolo/core/site.hpp>

#include <catch2/catch_test_macros.hpp>

using reticolo::Indexing;
using reticolo::Parity;
using reticolo::Site;

TEST_CASE("Indexing reports basic shape geometry", "[indexing]") {
    auto idx = Indexing::acquire({4, 4, 4, 4});
    REQUIRE(idx->ndims() == 4);
    REQUIRE(idx->nsites() == 256);
    REQUIRE(idx->shape() == Indexing::SizeVec{4, 4, 4, 4});
}

TEST_CASE("Periodic neighbours wrap on every direction", "[indexing]") {
    auto idx = Indexing::acquire({4, 4});  // d=2
    // Site 0 = (0,0). next in mu=0 => (1,0) = 1. prev in mu=0 wraps to (3,0) = 3.
    REQUIRE(idx->next(Site{0}, 0) == Site{1});
    REQUIRE(idx->prev(Site{0}, 0) == Site{3});
    // next in mu=1 from (0,0) => (0,1) = 4. prev wraps to (0,3) = 12.
    REQUIRE(idx->next(Site{0}, 1) == Site{4});
    REQUIRE(idx->prev(Site{0}, 1) == Site{12});
    // Two forward steps return to start on L=4.
    REQUIRE(idx->next(idx->next(idx->next(idx->next(Site{5}, 0), 0), 0), 0) == Site{5});
}

TEST_CASE("Parity matches sum-of-coords mod 2", "[indexing]") {
    auto idx = Indexing::acquire({4, 4});
    // (0,0): even. (1,0): odd. (0,1): odd. (1,1): even.
    REQUIRE(idx->parity_of(Site{0}) == Parity::Even);
    REQUIRE(idx->parity_of(Site{1}) == Parity::Odd);
    REQUIRE(idx->parity_of(Site{4}) == Parity::Odd);
    REQUIRE(idx->parity_of(Site{5}) == Parity::Even);
}

TEST_CASE("Even + odd sites partition the lattice", "[indexing]") {
    auto idx = Indexing::acquire({4, 4, 4, 4});
    REQUIRE(idx->even_sites().size() + idx->odd_sites().size() == idx->nsites());
    REQUIRE(idx->even_sites().size() == idx->nsites() / 2);
    REQUIRE(idx->odd_sites().size() == idx->nsites() / 2);

    for (Site s : idx->even_sites()) {
        REQUIRE(idx->parity_of(s) == Parity::Even);
    }
    for (Site s : idx->odd_sites()) {
        REQUIRE(idx->parity_of(s) == Parity::Odd);
    }
}

TEST_CASE("Indexing rejects empty shape and zero dimension", "[indexing]") {
    REQUIRE_THROWS_AS(Indexing::acquire({}), std::invalid_argument);
    REQUIRE_THROWS_AS(Indexing::acquire({4, 0, 4}), std::invalid_argument);
}
