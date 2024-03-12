/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: apps/LLRBoseGas.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>


    LLR workspace folder structure:
        OutPath
        ├── llr
        │   ├── logs
        │   │   ├── llr_worker[000].log
        │   │   ├── ...
        │   │   └── llr_worker[###].log
        │   └── meas
        │       ├── llr.h5
        │       ├── llr_worker[000].h5
        │       ├── ...
        │       └── llr_worker[###].h5
        └── reticolo.log
******************************************************************************/

#include <cstdlib>
#include <string>

#include "reticolo/action/RelativisticBoseGas.hpp"
#include "reticolo/lattice/lattice.hpp"
#include "reticolo/llr/_controller.hpp"
#include "reticolo/types/core.hpp"

using namespace reticolo;

auto main(int argc, char* argv[]) -> int {
    /* Define the lattice volume */
    intvect<4> Volume = {4, 4, 4, 4};

    /*  Set the output folder to be ./BoseGasLLR */
    std::string OutPath = "BoseGasLLR";

    /* Initialize the lattice */
    Lattice<ComplexD, 4> Lattice({4, 4, 4, 4});

    /* Initialize the action */
    action::RelativisticBoseGas::Params Par(1.0, 9.0, 0.0);

    /* Declare the LLRController
        This will take care of setting up and running the simulation
        Spawn al the LLRWorkers and control the multi-threaded operation */
    LLR::LLRController<action::RelativisticBoseGas> Cont(Par, OutPath, Volume, 48, 0.0025);

    /* Run the LLR simulation
        This will perform nMonteCarlo updates for each NR or RM step */
    Cont.run(50, 500, 1000, "BoseGasLLR", 8);

    return EXIT_SUCCESS;
}
