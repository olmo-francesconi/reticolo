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
#include "reticolo/lattice/lattice.hpp"
#include "reticolo/montecarlo/HMC.hpp"
#include "reticolo/types/core.hpp"

using namespace reticolo;

auto main(int argc, char* argv[]) -> int {
    /*  Set the output folder to be ./MetropolisMonteCarlo */
    std::string OutPath = "MetropolisMonteCarlo_out";

    /* Initialize the lattice */
    Lattice<ComplexD, 4> Lattice({8, 8, 8, 8});

    /* Initialize the action */
    action::RelativisticBoseGas::Params Par(1.0, 9.0, 0.0);
    action::RelativisticBoseGas         Action(Lattice, Par);

    // simulation workflow for indefinite end
    montecarlo::HMC Worker("HMC", Action, Lattice, 0, OutPath);

    Worker.run(100000, 1000, 10);

    return EXIT_SUCCESS;
}
