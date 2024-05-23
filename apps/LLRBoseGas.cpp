/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: apps/LLRBoseGas.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>


    LLR workspace folder structure:
        OutPath
        ├── raw_data
        │   ├── RunName[0]
        │   │   ├── logs
        │   │   │   ├── llr_worker[000].log
        │   │   │   ├── ...
        │   │   │   └── llr_worker[###].log
        │   │   └── measurements
        │   │       ├── llr_worker[000].h5
        │   │       ├── ...
        │   │       └── llr_worker[###].h5
        │   ├── ...
        │   └── RunName[###]
        ├── controller.log
        └── llr.h5
******************************************************************************/

#include <cstdlib>
#include <format>
#include <string>

#include "reticolo/action/RelativisticBoseGas.hpp"
#include "reticolo/llr/controller.hpp"
#include "reticolo/types/core.hpp"

using namespace reticolo;

auto main(int argc, char* argv[]) -> int {
    /* Define the lattice volume */
    intvect<4> Volume = {8, 8, 8, 8};
    // intvect<4> Volume = {4, 4, 4, 4};

    /*  Set the output folder */
    // std::string OutPath = std::format("/Volumes/Extreme SSD/Physics/LLR/BoseGas_4x4x4x4/{}/", argv[1]);
    // std::string OutPath = std::format("/Users/olmo/Desktop/bose_gas_llr/4_4_4_4/{}/", argv[1]);
    std::string OutPath = std::format("/Users/olmo/Desktop/bose_gas_llr/8_8_8_8/{}/", argv[2]);

    /* Initialize the action parameters */
    action::RelativisticBoseGas::Params Par(1.0, 9.0, std::stod(argv[2]));

    /* Declare the LLRController
        This will take care of setting up and running the simulation
        Spawn al the LLRWorkers and control the multi-threaded operation */
    LLR::LLRController<action::RelativisticBoseGas> Cont(Par, OutPath, Volume, 64, 0.00125);

    /* Run the LLR simulation
        This will perform nMonteCarlo updates for each NR or RM step */
    // for (int Rep = 0; Rep < 10; Rep++) {
    // std::string RunName = std::format("replica_{}", Rep);
    std::string RunName = argv[1];
    bool        SaveData = false;
    bool        SaveConfig = false;
    Cont.run(RunName, 100, 1000, 2000, SaveData, SaveConfig);
    // }

    return EXIT_SUCCESS;
}