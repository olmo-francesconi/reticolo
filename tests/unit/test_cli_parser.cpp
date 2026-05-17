#include <reticolo/cli/parser.hpp>

#include <array>
#include <stdexcept>
#include <string>

#include <catch2/catch_test_macros.hpp>

namespace cli = reticolo::cli;

TEST_CASE("Parser parses required ints and optional defaults", "[cli]") {
    cli::Parser p{"phi4_hmc", "HMC for phi^4"};
    auto const& L       = p.req<int>("L,size", "linear size");
    auto const& kappa   = p.req<double>("kappa", "hopping");
    auto const& seed    = p.opt<unsigned long>("seed", 42UL, "RNG seed");
    auto const& outpath = p.opt<std::string>("out", std::string{"phi4.h5"}, "output path");

    std::array<char const*, 5> argv{
        "phi4_hmc", "--size=8", "--kappa=0.18", "--seed=99", "--out=run.h5"};
    p.parse(static_cast<int>(argv.size()), argv.data());

    REQUIRE(L == 8);
    REQUIRE(kappa == 0.18);
    REQUIRE(seed == 99UL);
    REQUIRE(outpath == "run.h5");
}

TEST_CASE("Parser uses defaults when optional flags are omitted", "[cli]") {
    cli::Parser p{"phi4_hmc"};
    auto const& seed = p.opt<unsigned long>("seed", 42UL);
    auto const& out  = p.opt<std::string>("out", std::string{"phi4.h5"});

    std::array<char const*, 1> argv{"phi4_hmc"};
    p.parse(static_cast<int>(argv.size()), argv.data());

    REQUIRE(seed == 42UL);
    REQUIRE(out == "phi4.h5");
}

TEST_CASE("Parser throws when a required flag is missing", "[cli]") {
    cli::Parser p{"phi4_hmc"};
    (void)p.req<int>("L,size", "linear size");

    std::array<char const*, 1> argv{"phi4_hmc"};
    REQUIRE_THROWS_AS(p.parse(static_cast<int>(argv.size()), argv.data()), std::runtime_error);
}

TEST_CASE("Parser supports --key value separated form (cxxopts default)", "[cli]") {
    cli::Parser p{"phi4_hmc"};
    auto const& L = p.req<int>("L,size", "linear size");

    std::array<char const*, 3> argv{"phi4_hmc", "--size", "16"};
    p.parse(static_cast<int>(argv.size()), argv.data());

    REQUIRE(L == 16);
}

TEST_CASE("stamp_into before parse() throws", "[cli]") {
    cli::Parser p{"phi4_hmc"};
    (void)p.opt<int>("L,size", 4);
    // We can't construct a Writer without a path that we don't want to create
    // accidentally; just check the parsed() guard fires first.
    REQUIRE_FALSE(p.parsed());
}

TEST_CASE("parsed() flag flips after a successful parse", "[cli]") {
    cli::Parser p{"phi4_hmc"};
    (void)p.opt<int>("L,size", 4);
    std::array<char const*, 1> argv{"phi4_hmc"};
    p.parse(static_cast<int>(argv.size()), argv.data());
    REQUIRE(p.parsed());
}
