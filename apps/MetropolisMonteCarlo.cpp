/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: apps/MetropoliMonteCarlo.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>


    Montecarlo workspace folder structure:
        OutPath
        ├── meas
        │   └── Run_name.h5
        └── reticolo.log
******************************************************************************/

#include <cstdlib>
#include <iostream>
#include <string>

#include "reticolo/reticolo.hpp"
#include "reticolo/types/core.hpp"

using namespace reticolo;

auto main(int argc, char* argv[]) -> int {
    /* Define the lattice volume */
    uintvect<4> Volume = {4, 4, 4, 4};

    /*  Set the output folder to be ./MetropolisMonteCarlo */
    std::string OutPath = "MetropolisMonteCarlo";

    /* Initialize the action */
    action::BoseGas<ComplexD, ComplexD> Action(1.0, 9.0, 1.0);

    // simulation workflow for indefinite end

    return EXIT_SUCCESS;
}