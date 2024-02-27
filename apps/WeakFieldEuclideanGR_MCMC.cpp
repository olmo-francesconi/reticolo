/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: apps/MetropoliMonteCarlo.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

    Montecarlo workspace folder structure:
        WeakFieldEuclideanGR
        ├── meas
        │   └── WeakFieldEuclideanGR.h5
        └── reticolo.log
******************************************************************************/

#include <cstdlib>
#include <string>

#include "reticolo/action/WeakFieldEuclideanGR.hpp"
#include "reticolo/montecarlo/metropolis.hpp"
#include "reticolo/types/core.hpp"

using namespace reticolo;

auto main(int argc, char* argv[]) -> int {
    /* Define the lattice volume */
    uintvect<4> Volume = {6, 6, 6, 6};

    // /*  Set the output folder to be ./MetropolisMonteCarlo */
    std::string OutPath = "WeakFieldEuclideanGR";

    /* Initialize the action */
    action::WeakFieldEuclideanGR Action(5.7);

    // simulation workflow for indefinite end
    montecarlo::MetropolisWorker Worker("WeakFieldEuclideanGR", Action, Volume, 0, OutPath);

    Worker.run(10000, 0, 1, false, 1.0, false);

    return EXIT_SUCCESS;
}