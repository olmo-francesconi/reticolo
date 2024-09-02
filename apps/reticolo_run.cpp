#include <cstdlib>

#include "reticolo/Reticolo.hpp"
#include "reticolo/action/RelativisticBoseGas.hpp"
#include "reticolo/montecarlo/HMC.hpp"

using namespace reticolo;

auto main(int argc, char* argv[]) -> int {
    montecarlo::HMC<action::RelativisticBoseGas> a;

    reticolo_init(argc, argv);

    reticolo_run();

    reticolo_end();
};