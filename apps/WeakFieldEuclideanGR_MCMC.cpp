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
#include <sys/types.h>

#include <cstdlib>
#include <format>
#include <string>

#include "reticolo/action/WeakFieldEuclideanGR.hpp"
#include "reticolo/lattice/lattice.hpp"
#include "reticolo/montecarlo/Metropolis.hpp"
#include "reticolo/types/core.hpp"
#include "reticolo/types/hfield.hpp"

using namespace reticolo;

auto main(int argc, char* argv[]) -> int {
    /* Parse command line arguments */
    uint NUpdates = std::stoi(argv[1]);
    int  Tmax = std::stoi(argv[2]);
    int  Lmax = std::stoi(argv[3]);

    /* Set the output folder and run name*/
    std::string OutPath = std::format("LQGR_metropolis", Tmax, Lmax);
    std::string RunName = std::format("{0}x{1}^3", Tmax, Lmax);

    /* Set up Lattice */
    intvect<4>                Volume = {Tmax, Lmax, Lmax, Lmax};
    Lattice<HField<RealD>, 4> Lattice(Volume);

    /* Set up Action */
    action::WeakFieldEuclideanGR::Params Par(5.7);
    action::WeakFieldEuclideanGR         Action(Lattice, Par);

    /* Set up and run Monte Carlo */
    montecarlo::Metropolis MetropolisWorker(RunName, Action, Lattice, 0, OutPath, true, false, false);
    MetropolisWorker.setParams(0.1);
    MetropolisWorker.run("LQGR", NUpdates, 0, 1);

    return EXIT_SUCCESS;
}