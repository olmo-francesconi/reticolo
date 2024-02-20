/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: apps/llr.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#include <cstdlib>
#include <string>

#include "reticolo/reticolo.hpp"
#include "reticolo/types/core.hpp"

using namespace reticolo;

auto main(int argc, char* argv[]) -> int {
    /* Define the lattice volume */
    uintvect<4> Volume = {4, 4, 4, 4};

    /*  Set the output folder to be "./nt_nx_ny_nz/"
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
            └── reticolo.log                         */
    std::string OutPath = "BoseGasLLR";

    /* Initialize the action */
    action::BoseGas<ComplexD, ComplexD> Action(1.0, 9.0, 1.0);

    /* Declare the LLRController
        This will take care of setting up and running the simulation
        Spawn al the LLRWorkers and control the multi-threaded operation */
    LLR::LLRController Cont(Action);

    /* Initialize the LLRController
        This will actually allocate memory and set up all the workers */
    Cont.init(OutPath, Volume, 1, 0.0025);

    /* Run the LLR simulation
        This will perform nMonteCarlo updates for each NR or RM step */
    Cont.run(50, 50, 1000, "BoseGasLLR", 2);

    return EXIT_SUCCESS;
}
