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

#include <omp.h>

#include <cstdlib>
#include <exception>
#include <format>
#include <iostream>
#include <string>
#include <vector>

#include "reticolo/action/RelativisticBoseGas.hpp"
// #include "reticolo/action/WeakFieldEuclideanGR.hpp"
#include "reticolo/action/WeakFieldEuclideanGR.hpp"
#include "reticolo/lattice/lattice.hpp"
#include "reticolo/montecarlo/HMC.hpp"
// #include "reticolo/montecarlo/Metropolis.hpp"
#include "reticolo/montecarlo/Metropolis.hpp"
#include "reticolo/montecarlo/MonteCarloData.hpp"
#include "reticolo/tools/io_utils.hpp"
#include "reticolo/tools/timer.hpp"
#include "reticolo/types/core.hpp"
#include "reticolo/types/hfield.hpp"

using namespace reticolo;

auto main(int argc, char* argv[]) -> int {
    /*  Set the output folder */
    std::string OutPath = "test";

    uint NUpdates = std::stoi(argv[1]);

#pragma omp parallel num_threads(2)
#pragma omp          single
    {
#pragma omp task
        {
#pragma omp critical
            std::cout << "hmc starting..\n";

            Timer                               T;
            std::string                         RunName = std::format("HMC");
            Lattice<ComplexD, 4>                Lattice({16, 16, 16, 16});
            action::RelativisticBoseGas::Params Par(1.0, 9.0, 0.0);
            action::RelativisticBoseGas         Action(Lattice, Par);
            montecarlo::HMC                     HMCWorker(RunName, Action, Lattice, 0, OutPath);
            HMCWorker.setParams(20, 0.01);
            HMCWorker.run(NUpdates, 100, 1, true, true, 1.0, false);
#pragma omp critical
            std::cout << std::format("hmc done! [{:.2f} s]\n", T.elapsed_s());
        }
#pragma omp task
        {
#pragma omp critical
            std::cout << "met starting..\n";

            Timer                               T;
            std::string                         RunName = std::format("Met");
            Lattice<ComplexD, 4>                Lattice({16, 16, 16, 16});
            action::RelativisticBoseGas::Params Par(1.0, 9.0, 0.0);
            action::RelativisticBoseGas         Action(Lattice, Par);
            montecarlo::Metropolis              MetropolisWorker(RunName, Action, Lattice, 0, OutPath);
            MetropolisWorker.setParams(0.5);
            MetropolisWorker.run(NUpdates, 100, 1, true, true, 0.1, false);
#pragma omp critical
            std::cout << std::format("met done! [{:.2f} s]\n", T.elapsed_s());
        }
#pragma omp taskwait
    }
    return EXIT_SUCCESS;
}