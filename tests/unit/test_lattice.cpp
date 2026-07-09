#include <reticolo/core/indexing.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/core/site.hpp>

#include <algorithm>

#include <catch2/catch_test_macros.hpp>

using reticolo::Lattice;
using reticolo::Site;

TEST_CASE("Lattice default-constructs to T{} on every site", "[lattice]") {
    Lattice<double> const phi{{4, 4, 4}};
    REQUIRE(phi.nsites() == 64);
    REQUIRE(phi.ndims() == 3);
    for (Site s : phi.sites()) {
        REQUIRE(phi[s] == 0.0);
    }
}

TEST_CASE("Fill-construct seeds every site", "[lattice]") {
    Lattice<int> const phi{{4, 4}, 7};
    for (Site s : phi.sites()) {
        REQUIRE(phi[s] == 7);
    }
}

TEST_CASE("Element write/read is direct-indexed by Site", "[lattice]") {
    Lattice<double> phi{{4, 4}};
    phi[Site{5}] = 2.5;
    REQUIRE(phi[Site{5}] == 2.5);
    REQUIRE(phi[Site{0}] == 0.0);
    REQUIRE(phi[Site{15}] == 0.0);
}

TEST_CASE("Copy-construct deep-copies data but shares Indexing", "[lattice]") {
    Lattice<double> phi{{4, 4}};
    phi[Site{2}] = 11.0;

    Lattice<double> psi = phi;  // NOLINT(performance-unnecessary-copy-initialization)
    REQUIRE(psi[Site{2}] == 11.0);

    psi[Site{2}] = 99.0;
    REQUIRE(phi[Site{2}] == 11.0);
    REQUIRE(psi[Site{2}] == 99.0);

    REQUIRE(phi.indexing().get() == psi.indexing().get());
}

TEST_CASE("Sibling Lattice constructor reuses the source's Indexing", "[lattice]") {
    Lattice<double> phi{{4, 4, 4, 4}};
    Lattice<double> mom{phi.indexing()};

    REQUIRE(mom.nsites() == phi.nsites());
    REQUIRE(mom.indexing().get() == phi.indexing().get());
    for (Site s : mom.sites()) {
        REQUIRE(mom[s] == 0.0);
    }
}

TEST_CASE("Topology queries delegate to Indexing", "[lattice]") {
    Lattice<int> const phi{{4, 4}};
    REQUIRE(phi.next(Site{0}, 0) == Site{1});
    REQUIRE(phi.prev(Site{0}, 0) == Site{3});
    REQUIRE(phi.next(Site{3}, 0) == Site{0});   // +x wraps
    REQUIRE(phi.next(Site{0}, 1) == Site{4});   // +y = +stride
    REQUIRE(phi.prev(Site{0}, 1) == Site{12});  // -y wraps
}

TEST_CASE("sites() iota-view yields every Site exactly once in order", "[lattice]") {
    Lattice<int> const phi{{3, 3}};
    std::size_t expected = 0;
    for (Site s : phi.sites()) {
        REQUIRE(s.value() == expected);
        ++expected;
    }
    REQUIRE(expected == phi.nsites());
}

TEST_CASE("data() exposes a contiguous flat buffer", "[lattice]") {
    Lattice<double> phi{{4, 4}};
    phi[Site{0}] = 1.0;
    phi[Site{1}] = 2.0;
    REQUIRE(phi.data()[0] == 1.0);
    REQUIRE(phi.data()[1] == 2.0);
    REQUIRE(std::count(phi.begin(), phi.end(), 0.0) == 14);
}
