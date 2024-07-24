/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: apps/BoseGas_HMCMonteCarlo.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

******************************************************************************/

#include <format>
#include <iostream>
#include <string>
#include <vector>

#include "reticolo/action/RelativisticBoseGas.hpp"
#include "reticolo/lattice/lattice.hpp"
#include "reticolo/montecarlo/HMC.hpp"
#include "reticolo/types/core.hpp"

using namespace reticolo;

auto main(int argc, char* argv[]) -> int {
    std::vector<RealD> TrajLengths = {0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5};

#pragma omp parallel for num_threads(6)
    for (auto TrajLen : TrajLengths) {
        /* Set the output folder */
        std::string OutPath = argv[1];

        /* Initialize the lattice */
        Lattice<ComplexD, 4> Lattice({12, 12, 12, 12});

        /* Initialize the action */
        action::RelativisticBoseGas::Params Par(1.0, 9.0, 0.0);
        action::RelativisticBoseGas         Action(Lattice, Par);

        montecarlo::HMC MonteCarloWorker(std::format("HMC_{:.3f}", TrajLen),  // Worker name
                                         Action,                              // Action
                                         Lattice,                             // Lattice
                                         0,                                   // RNG seed
                                         OutPath,                             // Output directory
                                         false,                               // Print to stdout
                                         true,                                // Save measurements
                                         false);                              // Save configs

        std::vector<int> Steps = {1, 2, 5, 10, 20, 50};

        for (auto StepNum : Steps) {
#pragma omp critical
            std::cout << std::format("starting : {:>8.4f} - {:>2} \n", TrajLen, StepNum);

            MonteCarloWorker.setParams(TrajLen, StepNum);

            MonteCarloWorker.run(std::format("hmc_{}", StepNum),  // Run name
                                 std::stoi(argv[2]),  // Total Monte Carlo Updates (0=infinite, stop with CTRL+C)
                                 0,                   // Initial thermalization steps
                                 1,                   // Measure interval
                                 true,                // Field initialization
                                 false);              // Hot start (random field initialization)
        }
    }
}
