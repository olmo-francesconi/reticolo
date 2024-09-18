/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: Reticolo.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>

#include "cxxopts.hpp"
#include "reticolo/core/tools/io_utils.hpp"
#include "reticolo/core/tools/logger.hpp"
#include "reticolo/modules/factory/ModuleBase.hpp"
#include "reticolo/modules/factory/ModuleFactory.hpp"
#include "reticolo/runtime/ReticoloCore.hpp"
#include "yaml-cpp/exceptions.h"
#include "yaml-cpp/node/node.h"

namespace reticolo {

/*--------------------------------------------------------------------------------------------------
  reticolo_init()
--------------------------------------------------------------------------------------------------*/
inline void reticolo_init(int argc, char* argv[]) {
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
inline void reticolo_run() {
    /* For each key in Setup["workflows"] setup and run the simulation */
    auto ToDo = (ReticoloCore::getSetup())["workflows"];

    std::string ModuleName;
    YAML::Node  ModuleConfig;
    for (const auto& Wrkflw : ToDo) {
        try {
            /* Check that there is only one top-level key */
            ModuleName = Wrkflw.begin()->first.as<std::string>();
            ModuleConfig = Wrkflw.begin()->second;

            auto Module = ModuleFactory::MakeModule(ModuleName, ModuleConfig["action"]["name"].as<std::string>());

            /* Configure the module */
            Module->setup(ModuleConfig);

            /* Run the simulations */
            for (const auto& RunConfig : ModuleConfig["runs"]) {
                Module->execute(RunConfig);
            }

        } catch (const YAML::Exception& e) {
            IO::GlobalLogger << IO::LI_erro() + "Failed to parse configuration of module: " + ModuleName + " [" +
                                    ModuleConfig["name"].as<std::string>() + "]\n";
            IO::GlobalLogger << IO::LI_erro() + e.what() + "\n";
            exit(EXIT_FAILURE);
        } catch (const std::exception& e) {
            IO::GlobalLogger << IO::LI_erro() + e.what() + "\n";
            exit(EXIT_FAILURE);
        }
    }
};

/*--------------------------------------------------------------------------------------------------
  reticolo_end()
--------------------------------------------------------------------------------------------------*/
inline void reticolo_end() {
    /* General clean-up */
};

}  // namespace reticolo
