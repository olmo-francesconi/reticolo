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
#include <format>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "reticolo/action/RelativisticBoseGas.hpp"
#include "reticolo/action/WeakFieldEuclideanGR.hpp"
#include "reticolo/lattice/lattice.hpp"
#include "reticolo/montecarlo/Metropolis.hpp"
#include "reticolo/montecarlo/montecarlo.hpp"
#include "reticolo/types/core.hpp"

using namespace reticolo;

auto main(int argc, char* argv[]) -> int {
    /* Define the lattice volume */
    uintvect<4> Volume = {4, 4, 4, 4};

    /*  Set the output folder */
    std::string OutPath = "test";

    /* Initialize the action */
    action::WeakFieldEuclideanGR Action(5.7);
    // action::RelativisticBoseGas<ComplexD, ComplexD> Action(1.0, 9.0, 0.0);

    std::string     RunName = std::format("HMC");
    montecarlo::HMC HMCWorker(RunName, Action, Volume, 0, OutPath);
    HMCWorker.initialize(2, 0.01);
    HMCWorker.run(1, 0, 1, false, 0.1, false);
    // montecarlo::Metropolis MetropolisWorker("Metropolis", Action, Volume, 0, OutPath);
    // // MetropolisWorker.initialize(10, 0.001);
    // MetropolisWorker.run(1000000, 0, 1, false, 0.1, false);

    return EXIT_SUCCESS;
}
