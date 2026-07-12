// log::off() so construction auto-announces don't drown test output.
// Tests that need log output (test_log.cpp) call log::on() locally.

#include <reticolo/core/log/log.hpp>

#include <catch2/catch_session.hpp>

int main(int argc, char* argv[]) {
    reticolo::log::off();
    return Catch::Session().run(argc, argv);
}
