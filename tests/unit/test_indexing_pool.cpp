#include <reticolo/core/indexing.hpp>

#include <catch2/catch_test_macros.hpp>

using reticolo::Indexing;

TEST_CASE("Identical shape requests share one Indexing instance", "[indexing][pool]") {
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

TEST_CASE("Pool entries are reclaimed when their last shared_ptr expires", "[indexing][pool]") {
    Indexing const* raw = nullptr;
    {
        auto a = Indexing::acquire({16, 16});
        raw    = a.get();
        REQUIRE(a.use_count() == 1);
    }
    auto b = Indexing::acquire({16, 16});
    REQUIRE(b.use_count() == 1);
    (void)raw;
}
