#include "reticolo/Reticolo.hpp"

using namespace reticolo;

auto main(int argc, char* argv[]) -> int {
    reticolo_init(argc, argv);

    reticolo_run();

    reticolo_end();
};