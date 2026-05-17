#include <reticolo/core/bc.hpp>
#include <reticolo/core/indexing.hpp>

#include <catch2/catch_test_macros.hpp>

using reticolo::Bc;
using reticolo::BcMask;
using reticolo::Indexing;

TEST_CASE("Identical (shape, bcs) requests share one Indexing instance", "[indexing][pool]") {
    auto a = Indexing::acquire({8, 8, 8, 8});
    auto b = Indexing::acquire({8, 8, 8, 8});
    REQUIRE(a.get() == b.get());
    REQUIRE(a.use_count() >= 2);
}

TEST_CASE("Different shapes get distinct Indexing instances", "[indexing][pool]") {
    auto a = Indexing::acquire({8, 8, 8, 8});
    auto b = Indexing::acquire({8, 8, 8, 4});
    REQUIRE(a.get() != b.get());
}

TEST_CASE("Different BCs produce distinct Indexing instances", "[indexing][pool]") {
    auto a = Indexing::acquire({8, 8}, BcMask{Bc::Periodic, Bc::Periodic});
    auto b = Indexing::acquire({8, 8}, BcMask{Bc::Open, Bc::Periodic});
    REQUIRE(a.get() != b.get());
}

TEST_CASE("Default BcMask is promoted to all-periodic at the requested ndims", "[indexing][pool]") {
    auto a = Indexing::acquire({4, 4, 4});
    auto c = Indexing::acquire({4, 4, 4}, BcMask{3, Bc::Periodic});
    REQUIRE(a.get() == c.get());
}

TEST_CASE("Pool entries are reclaimed when their last shared_ptr expires", "[indexing][pool]") {
    Indexing const* raw = nullptr;
    {
        auto a = Indexing::acquire({16, 16});
        raw    = a.get();
        REQUIRE(a.use_count() == 1);
    }
    // After `a` goes out of scope, the next acquire of the same shape
    // must build a fresh instance — the previous weak_ptr expired.
    auto b = Indexing::acquire({16, 16});
    // Best-effort check: the new pointer is fresh. (Same address is possible if
    // the allocator reuses storage, so check that use_count restarts at 1.)
    REQUIRE(b.use_count() == 1);
    (void)raw;
}
