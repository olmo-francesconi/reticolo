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
#include <string>

#include "reticolo/action/RelativisticBoseGas.hpp"
#include "reticolo/montecarlo/metropolis.hpp"
#include "reticolo/types/core.hpp"

using namespace reticolo;

auto main(int argc, char* argv[]) -> int {
    /* Define the lattice volume */
    uintvect<4> Volume = {4, 4, 4, 4};

    // Lattice<RealD, 4> Lattice(Volume);

    /*  Set the output folder to be ./MetropolisMonteCarlo */
    std::string OutPath = "MetropolisMonteCarlo_out";

    /* Initialize the action */
    action::RelativisticBoseGas<ComplexD, ComplexD> Action(1.0, 9.0, 1.0);

    // simulation workflow for indefinite end
    montecarlo::MetropolisWorker Worker(Action);

    Worker.init(Volume, "MetropolisMonteCarlo", 0, OutPath);

    Worker.run(100000, 1000, 1);

    return EXIT_SUCCESS;
}