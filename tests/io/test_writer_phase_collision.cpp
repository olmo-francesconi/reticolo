#include <reticolo/io/writer.hpp>

#include "../test_helpers.hpp"

#include <stdexcept>
#include <string>

#include <catch2/catch_test_macros.hpp>

using reticolo::io::Writer;

TEST_CASE("start_phase succeeds the first time and throws on reuse", "[io][phase]") {
    reticolo::test::ScratchH5 f{"phase_collision"};

    Writer w{f.path()};
    REQUIRE_NOTHROW(w.start_phase("therm"));
    REQUIRE_NOTHROW(w.start_phase("prod"));
    REQUIRE_THROWS_AS(w.start_phase("therm"), std::runtime_error);
}

TEST_CASE("start_phase rejects names whose root group already exists in the file", "[io][phase]") {
    reticolo::test::ScratchH5 f{"phase_preexisting"};

    Writer w{f.path()};
    // Create /prod by writing a series under it first.
    {
        auto s = w.series<double>("/prod/obs/s");
        s.append(1.0);
    }
    REQUIRE_THROWS_AS(w.start_phase("prod"), std::runtime_error);
}
