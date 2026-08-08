#include <reticolo/core/field/lattice.hpp>
#include <reticolo/core/field/site.hpp>

#include <algorithm>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using reticolo::Lattice;
using reticolo::Site;

// Lattice geometry — now the field's OWN state (impl::LatticeGeometry base),
// not a pooled `Indexing` object reached through a shared_ptr. The behaviour it
// has to keep is exactly what the old Indexing tests pinned; what changes is
// that it is exercised through the field, because that is the only way it is
// reachable now.

TEST_CASE("a field reports its own shape geometry", "[core][geometry]") {
    Lattice<double> const l{{4, 4, 4, 4}};
    REQUIRE(l.ndims() == 4);
    REQUIRE(l.nsites() == 256);
    REQUIRE(std::ranges::equal(l.shape(), std::vector<std::size_t>{4, 4, 4, 4}));
}

TEST_CASE("periodic neighbours wrap on every direction", "[core][geometry]") {
    Lattice<double> const l{{4, 4}};  // d=2
    // Site 0 = (0,0). next in mu=0 => (1,0) = 1. prev in mu=0 wraps to (3,0) = 3.
    REQUIRE(l.next(Site{0}, 0) == Site{1});
    REQUIRE(l.prev(Site{0}, 0) == Site{3});
    // next in mu=1 from (0,0) => (0,1) = 4. prev wraps to (0,3) = 12.
    REQUIRE(l.next(Site{0}, 1) == Site{4});
    REQUIRE(l.prev(Site{0}, 1) == Site{12});
    // Two forward steps return to start on L=4.
    REQUIRE(l.next(l.next(l.next(l.next(Site{5}, 0), 0), 0), 0) == Site{5});
}

TEST_CASE("packed strides drive next/prev on a non-cubic shape", "[core][geometry]") {
    // strides() is gone (its only caller was its own test); the packed layout is
    // observable through the neighbour walk instead, which is what it existed for.
    // shape {4,5,6} => stride {1,4,20}.
    Lattice<double> const l{{4, 5, 6}};
    REQUIRE(l.nsites() == 120);
    REQUIRE(l.next(Site{0}, 0) == Site{1});   // stride 1
    REQUIRE(l.next(Site{0}, 1) == Site{4});   // stride 4
    REQUIRE(l.next(Site{0}, 2) == Site{20});  // stride 20
}

TEST_CASE("a field rejects an empty, zero or over-wide shape", "[core][geometry]") {
    REQUIRE_THROWS_AS(Lattice<double>{std::vector<std::size_t>{}}, std::invalid_argument);
    REQUIRE_THROWS_AS((Lattice<double>{{4, 0, 4}}), std::invalid_argument);
    REQUIRE_THROWS_AS((Lattice<double>{{2, 2, 2, 2, 2}}), std::invalid_argument);
}

TEST_CASE("sibling fields are built from a shape, not a shared handle", "[core][geometry]") {
    // The old pool made two same-shape fields point at ONE Indexing, and the
    // tests asserted that by pointer identity. There is no shared object now:
    // the geometry is inline, so the property that actually matters is that the
    // extents agree — and a sibling costs four multiplies to rebuild.
    Lattice<double> const phi{{6, 6, 6}};
    Lattice<double> const mom{phi.shape()};

    REQUIRE(mom.nsites() == phi.nsites());
    REQUIRE(mom.ndims() == phi.ndims());
    REQUIRE(std::ranges::equal(mom.shape(), phi.shape()));
    REQUIRE(mom.same_geometry(phi));

    Lattice<double> const other{{6, 6, 7}};
    REQUIRE_FALSE(other.same_geometry(phi));

    // No heap indirection left: the geometry travels with the field by value.
    STATIC_REQUIRE(std::is_trivially_copyable_v<reticolo::impl::LatticeGeometry>);
}
