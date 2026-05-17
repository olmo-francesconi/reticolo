#include <reticolo/core/log.hpp>

#include <sstream>
#include <string>

#include <catch2/catch_test_macros.hpp>

namespace log = reticolo::log;

TEST_CASE("log::info/warn/error write to the redirected stream", "[log]") {
    std::ostringstream os;
    log::set_stream(os);

    log::info("hello");
    log::warn("careful");
    log::error("oops");

    std::string const out = os.str();
    REQUIRE(out.find("[i] hello") != std::string::npos);
    REQUIRE(out.find("[!] careful") != std::string::npos);
    REQUIRE(out.find("[E] oops") != std::string::npos);

    log::set_stream(std::clog);
}

TEST_CASE("log::Section frames a block and nested sections indent", "[log]") {
    std::ostringstream os;
    log::set_stream(os);

    {
        log::Section outer{"outer"};
        log::info("inside outer");
        {
            log::Section inner{"inner"};
            log::info("inside inner");
        }
    }

    std::string const out = os.str();
    REQUIRE(out.find("+ outer") != std::string::npos);
    REQUIRE(out.find("+ inner") != std::string::npos);
    REQUIRE(out.find("- inner") != std::string::npos);
    REQUIRE(out.find("- outer") != std::string::npos);
    // Inside outer, depth becomes 1 → indent of 2 spaces.
    REQUIRE(out.find("  [i] inside outer") != std::string::npos);
    // Inside inner, depth becomes 2 → indent of 4 spaces.
    REQUIRE(out.find("    [i] inside inner") != std::string::npos);

    log::set_stream(std::clog);
}
