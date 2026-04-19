#include "reticolo/runtime/runtime.hpp"

using namespace reticolo;

auto main(int argc, char* argv[]) -> int {
    try {
        reticolo_init(argc, argv);

        reticolo_run();

        reticolo_end();
    } catch (const RuntimeExit& e) {
        if (std::string_view(e.what()).empty()) {
            return e.exit_code();
        }
        std::cout << e.what() << '\n';
        return e.exit_code();
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
};
