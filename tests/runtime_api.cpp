#include <cstdio>
#include <cstdlib>
#include <string_view>

#include "reticolo/runtime/runtime.hpp"

namespace {

void require(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "TEST FAILED: %s\n", msg);
        std::abort();
    }
}

}  // namespace

auto main() -> int {
    try {
        reticolo::ReticoloCore::readSetup("/definitely/not/a/real/reticolo-config.yaml");
        require(false, "ReticoloCore::readSetup should throw for a missing file");
    } catch (const std::runtime_error& e) {
        require(std::string_view(e.what()).find("Failed to load configuration") != std::string_view::npos,
                "missing-file error should mention configuration loading");
    }

    try {
        const char* argv[] = {"reticolo_run"};
        reticolo::reticolo_init(1, const_cast<char**>(argv));
        require(false, "reticolo_init should throw RuntimeExit when config is missing");
    } catch (const reticolo::RuntimeExit& e) {
        require(e.exit_code() == EXIT_SUCCESS, "missing-config RuntimeExit should preserve success exit code");
        require(std::string_view(e.what()).find("--config") != std::string_view::npos,
                "help output should mention --config");
    }

    return EXIT_SUCCESS;
}
