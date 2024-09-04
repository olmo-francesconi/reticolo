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
#include <vector>

#include "reticolo/action/RelativisticBoseGas.hpp"
#include "reticolo/llr/LLRController.hpp"

using namespace reticolo;

auto main(int argc, char* argv[]) -> int {
    /* Define the lattice volume */
    // intvect<4> Volume = {8, 8, 8, 8};
    // intvect<4> Volume = {6, 6, 6, 6};
    std::vector<int> Volume = {4, 4, 4, 4};

    /*  Set the output folder */
    // std::string OutPath = std::format("/Volumes/Extreme SSD/Physics/bose_gas_llr/8_8_8_8/{}/", argv[2]);
    // std::string OutPath = std::format("/Volumes/Extreme SSD/Physics/bose_gas_llr/6_6_6_6/{}/", argv[1]);
    std::string OutPath = std::format("/Volumes/Extreme SSD/Physics/bose_gas_llr/4_4_4_4_hmc/{}/", argv[1]);
    // std::string OutPath = std::format("/Users/olmo/Desktop/bose_gas_llr/4_4_4_4_hmcmet/{}/", argv[1]);
    // std::string OutPath = std::format("/Users/olmo/Desktop/bose_gas_llr/8_8_8_8/{}/", argv[2]);

    /* Initialize the action parameters */
    action::RelativisticBoseGas::Params Par(1.0, 9.0, std::stod(argv[1]));

    /* Declare the LLRController
        This will take care of setting up and running the simulation
        Spawn al the LLRWorkers and control the multi-threaded operation */
    LLR::LLRController<action::RelativisticBoseGas> Cont(Par, OutPath, Volume, 64, 0.001);

    /* Run the LLR simulation
        This will perform nMonteCarlo updates for each NR or RM step */
    for (int Rep = 0; Rep < 24; Rep++) {
        std::string RunName = std::format("replica_{}", Rep);
        // std::string RunName = std::format("{}", argv[1]);
        bool SaveData = true;
        bool SaveConfig = false;
        Cont.run(RunName, 10, 2000, 2000, SaveData, SaveConfig);
    }

    return EXIT_SUCCESS;
}
