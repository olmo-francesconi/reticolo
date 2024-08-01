/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: apps/MetropoliMonteCarlo.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

    Montecarlo workspace folder structure:
        WeakFieldEuclideanGR
        ├── logs
        │   └── RunName.h5
        ├── meas
        │   └── RunName.h5
        └── reticolo.log
******************************************************************************/

#include <omp.h>

#include <cstdlib>
#include <format>
#include <iostream>
#include <string>
#include <vector>

#include "reticolo/action/WeakFieldEuclideanGR.hpp"
#include "reticolo/lattice/lattice.hpp"
#include "reticolo/montecarlo/HMC.hpp"
#include "reticolo/montecarlo/Metropolis.hpp"
#include "reticolo/tools/timer.hpp"
#include "reticolo/types/core.hpp"
#include "reticolo/types/hfield.hpp"

using namespace reticolo;

auto main(int argc, char* argv[]) -> int {
    /*  Set the output folder */

    uint NUpdates = std::stoi(argv[1]);

    int        T = std::stoi(argv[2]);
    int        L = std::stoi(argv[3]);
    intvect<4> Volume = {T, L, L, L};

    std::string OutPath = std::format("WFGR_comp_{0}_{1}_{1}_{1}", T, L);

    std::vector<int>    Steps = {20};
    std::vector<double> StepSizes = {0.004, 0.002, 0.001, 0.0005, 0.00025, 0.000125};
    std::vector<double> MetStepSizes = {0.1};

#pragma omp parallel num_threads(4)
    {
        // #pragma omp for collapse(2) schedule(static, 1) nowait
        //         for (const auto& Step : Steps) {
        //             for (const auto& StepSize : StepSizes) {
        //                 std::string RunName = std::format("hmc_{}_{:f}", Step, StepSize);
        // #pragma omp critical
        //                 std::cout << "starting run: " << RunName << '\n';
        //                 Timer                                Timer;
        //                 Lattice<HField<RealD>, 4>            Lattice(Volume);
        //                 action::WeakFieldEuclideanGR::Params Par(5.7);
        //                 action::WeakFieldEuclideanGR         Action(Lattice, Par);
        //                 montecarlo::HMC                      HMCWorker(RunName, Action, Lattice, 0, OutPath);
        //                 HMCWorker.setParams(Step, StepSize);
        //                 HMCWorker.run(NUpdates, 0, 1);
        // #pragma omp critical
        //                 std::cout << "finished run: " << RunName << std::format(" [{:.2f} s]\n", Timer.elapsed_s());
        //             }
        //         }

        // #pragma omp for schedule(static, 1)
        //         for (const auto& StepSize : MetStepSizes) {
        //             std::string RunName = std::format("met_{:f}", StepSize);
        // #pragma omp critical
        //             std::cout << "starting run: " << RunName << '\n';
        //             Timer                                Timer;
        //             Lattice<HField<RealD>, 4>            Lattice(Volume);
        //             action::WeakFieldEuclideanGR::Params Par(5.7);
        //             action::WeakFieldEuclideanGR         Action(Lattice, Par);
        //             montecarlo::Metropolis               MetropolisWorker(RunName, Action, Lattice, 0, OutPath,
        //             true); MetropolisWorker.setParams(StepSize); MetropolisWorker.run(NUpdates, 0, 1);
        // #pragma omp critical
        //             std::cout << "finished run: " << RunName << std::format(" [{:.2f} s]\n", Timer.elapsed_s());
        //         }
    }

    return EXIT_SUCCESS;
}