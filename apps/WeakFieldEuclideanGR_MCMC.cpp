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
    uintvect<4> Volume = {4, 4, 4, 4};

    // Lattice<RealD, 4> Lattice(Volume);

    /*  Set the output folder to be ./MetropolisMonteCarlo */
    std::string OutPath = "WeakFieldEuclideanGR";

    /* Initialize the action */
    action::WeakFieldEuclideanGR<RealD, RealD> Action(1.0);

    // simulation workflow for indefinite end
    montecarlo::MetropolisWorker Worker(Action);

    Worker.init(Volume, "WeakFieldEuclideanGR", 0, OutPath);

    // Worker.run(1000, 0, 1);

    return EXIT_SUCCESS;
}