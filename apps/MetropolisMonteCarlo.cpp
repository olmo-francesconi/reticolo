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

// #include "reticolo/action/RelativisticBoseGas.hpp"
#include "reticolo/action/WeakFieldEuclideanGR.hpp"
#include "reticolo/lattice/lattice.hpp"
#include "reticolo/montecarlo/Metropolis.hpp"
#include "reticolo/types/core.hpp"
#include "reticolo/types/hfield.hpp"

using namespace reticolo;

auto main(int argc, char* argv[]) -> int {
    /*  Set the output folder to be ./MetropolisMonteCarlo */
    std::string OutPath = "MetropolisMonteCarlo_out";

    /* Initialize the lattice */
    Lattice<HField<RealD>, 4> Lattice({6, 6, 6, 6});

    /* Initialize the action */
    action::WeakFieldEuclideanGR::Params Par(5.7);
    action::WeakFieldEuclideanGR         Action(Lattice, Par);

    // simulation workflow for indefinite end
    montecarlo::Metropolis MetropolisWorker("RelativisticBoseGas_Worker",  // Worker name
                                            Action,                        // Action
                                            Lattice,                       // Lattice
                                            0,                             // RNG seed
                                            OutPath,                       // Output directory
                                            true,                          // Print to stdout
                                            true,                          // Save measurements
                                            false);                        // Save configs

    MetropolisWorker.run("mcmc", 0, 0, 1, true, false);

    return EXIT_SUCCESS;
}
