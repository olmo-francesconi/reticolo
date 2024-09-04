/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: Reticolo.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "cxxopts.hpp"
#include "reticolo/ReticoloCore.hpp"
#include "reticolo/modules/factory/ModuleBase.hpp"
#include "reticolo/modules/factory/ModuleFactory.hpp"
#include "reticolo/tools/io_utils.hpp"
#include "reticolo/tools/logger.hpp"
#include "yaml-cpp/exceptions.h"
#include "yaml-cpp/node/node.h"

namespace reticolo {

/*--------------------------------------------------------------------------------------------------
  reticolo_init()
--------------------------------------------------------------------------------------------------*/
void reticolo_init(int argc, char* argv[]) {
    std::cout << "DEBUG: reticolo_init()\n";
    /* Setup cxxopts arguments parser */
    cxxopts::Options Options("reticolo_run", "A thing that does stuff");
    Options.add_options()                                                       //
        ("config,c", "YAML configuration file", cxxopts::value<std::string>())  //
        ("h,help", "Print usage");
    auto Result = Options.parse(argc, argv);

    if (Result.count("help") != 0U) {
        std::cout << Options.help() << std::endl;
        exit(EXIT_SUCCESS);
    }

    std::string SetupFileName;
    if (Result.count("config") != 0U) {
        SetupFileName = Result["config"].as<std::string>();
    } else {
        std::cout << Options.help() << std::endl;
        exit(EXIT_SUCCESS);
    }

    std::cout << IO::pretty_welcome() << "\n";

    /* Initialize the ReticoloCore Singleton */
    ReticoloCore::Init();

    /* Read the setup file */
    ReticoloCore::readSetup(SetupFileName);

    /* Setup multithread operations */

    /* Initialize global variables */
    IO::GlobalLogger.init("./", "reticolo.log");

    /* Initialize global Logger */
};

/*--------------------------------------------------------------------------------------------------
  reticolo_run()
--------------------------------------------------------------------------------------------------*/
void reticolo_run() {
    std::cout << "DEBUG: reticolo_run()\n";

    auto ToDo = (ReticoloCore::getSetup())["workflows"];
    /* For each key in Setup["workflows"] setup and run the simulation */
    std::string ModuleName;
    YAML::Node  ModuleConfig;
    for (const auto& Wrkflw : ToDo) {
        try {
            ModuleName = Wrkflw.first.as<std::string>();
            ModuleConfig = Wrkflw.second;

            auto Module = ModuleFactory::MakeModule(ModuleName, ModuleConfig["action"]["name"].as<std::string>());

            /* Configure the module */
            Module->setup(ModuleConfig);

            for (const auto& RunConfig : ModuleConfig["runs"]) {
                Module->execute(RunConfig);
            }

        } catch (const YAML::Exception& e) {
            IO::GlobalLogger << IO::LI_erro() + "Failed to parse configuration of module: " + ModuleName + " [" +
                                    ModuleConfig["name"].as<std::string>() + "]\n";
            IO::GlobalLogger << IO::LI_erro() + e.what() + "\n";
        }
    }
};

/*--------------------------------------------------------------------------------------------------
  reticolo_end()
--------------------------------------------------------------------------------------------------*/
void reticolo_end() {
    std::cout << "DEBUG: reticolo_end()\n";
    /* General clean-up */
};
}  // namespace reticolo
