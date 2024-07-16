/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: apps/MetropoliMonteCarlo.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

******************************************************************************/

#include <cstdlib>
#include <string>

#include "reticolo/action/WeakFieldEuclideanGR.hpp"
#include "reticolo/lattice/lattice.hpp"
#include "reticolo/montecarlo/Metropolis.hpp"
#include "reticolo/types/core.hpp"
#include "reticolo/types/hfield.hpp"

using namespace reticolo;

auto main(int argc, char* argv[]) -> int {
    /*  Set the output folder to be ./MetropolisMonteCarlo_out */
    std::string OutPath = "MetropolisMonteCarlo_out";

    /* Initialize the lattice */
    Lattice<HField<RealD>, 4> Lattice({6, 6, 6, 6});

    /* Initialize the action */
    action::WeakFieldEuclideanGR::Params Par(5.7);
    action::WeakFieldEuclideanGR         Action(Lattice, Par);

    // simulation workflow for indefinite end
    montecarlo::Metropolis MetropolisWorker("LQGR",   // Worker name
                                            Action,   // Action
                                            Lattice,  // Lattice
                                            0,        // RNG seed
                                            OutPath,  // Output directory
                                            true,     // Print to stdout
                                            true,     // Save measurements
                                            false);   // Save configs

    MetropolisWorker.setParams(1.0);

    MetropolisWorker.run("mcmc",  // Run name
                         0,       // Total Monte Carlo Updates (0=infinite, stop with CTRL+C)
                         0,       // Initial thermalization steps
                         1,       // Measure interval
                         true,    // Field initialization
                         false);  // Hot start (random field initialization)

    return EXIT_SUCCESS;
}