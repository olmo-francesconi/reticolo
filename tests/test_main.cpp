// Shared Catch2 entry point for all reticolo unit / physics tests.
//
// Silences the logger before any test fixture runs — tests construct
// lattices, RNGs, actions, algorithms, writers, etc., and every one of
// those auto-announces on construction. Without this switch the test
// output would be drowned in setup chatter that has nothing to do with
// what's being asserted.
//
// Tests that *do* want to inspect log output (e.g. `test_log.cpp`) flip
// the switch back on explicitly inside their own TEST_CASE.

#include <reticolo/core/log.hpp>

#include <catch2/catch_session.hpp>

int main(int argc, char* argv[]) {
    reticolo::log::off();
    return Catch::Session().run(argc, argv);
}
