/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: apps/BoseGas_MetropolisMonteCarlo.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

******************************************************************************/

#include <format>
#include <string>

#include "reticolo/action/RelativisticBoseGas.hpp"
#include "reticolo/lattice/Lattice.hpp"
#include "reticolo/montecarlo/Metropolis.hpp"
#include "reticolo/types/core.hpp"

using namespace reticolo;

auto main(int argc, char* argv[]) -> int {
    /* Set the output folder */
    std::string OutPath = "BoseGas_Metropolis";

    /* Initialize the lattice */
    Lattice<ComplexD> Lattice({4, 4, 4, 4});

    /* Initialize the action */
    action::RelativisticBoseGas::Params Par(1.0, 9.0, 0.0);
    action::RelativisticBoseGas         Action(Lattice, Par);

    montecarlo::Metropolis MonteCarloWorker("Met_bose",  // Worker name
                                            Action,      // Action
                                            Lattice,     // Lattice
                                            0,           // RNG seed
                                            OutPath,     // Output directory
                                            true,        // Print to stdout
                                            true,        // Save measurements
                                            false);      // Save configs

    for (int Idx = 0; Idx < 1; Idx++) {
        RealD PropWidth = 0.1 * (Idx + 1);

        MonteCarloWorker.setParams(PropWidth);

        MonteCarloWorker.run(std::format("met_{:.1f}", PropWidth),  // Run name
                             std::stoi(argv[1]),  // Total Monte Carlo Updates (0=infinite, stop with CTRL+C)
                             0,                   // Initial thermalization steps
                             1,                   // Measure interval
                             true,                // Field initialization
                             false);              // Hot start (random field initialization)
    }
}
