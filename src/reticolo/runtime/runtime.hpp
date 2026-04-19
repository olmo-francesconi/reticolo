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
#include <stdexcept>
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

class RuntimeExit : public std::runtime_error {
  public:
    RuntimeExit(int exit_code, std::string message) : std::runtime_error(std::move(message)), _ExitCode(exit_code) {}

    [[nodiscard]] auto exit_code() const noexcept -> int { return _ExitCode; }

  private:
    int _ExitCode;
};

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

    if (Result.contains("help")) {
        throw RuntimeExit(EXIT_SUCCESS, Options.help());
    }

    std::string SetupFileName;
    if (Result.contains("config")) {
        SetupFileName = Result["config"].as<std::string>();
    } else {
        throw RuntimeExit(EXIT_SUCCESS, Options.help());
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
            const auto ActionName = ModuleConfig["action"]["name"].as<std::string>();

            ModuleFactory::ValidateModuleAction(ModuleName, ActionName);

            auto Module = ModuleFactory::MakeModule(ModuleName, ActionName);

            /* Configure the module */
            Module->setup(ModuleConfig);

            /* Run the simulations */
            for (const auto& RunConfig : ModuleConfig["runs"]) {
                Module->execute(RunConfig);
            }

        } catch (const YAML::Exception& e) {
            std::string Message = "Failed to parse configuration of module: " + ModuleName;
            if (ModuleConfig.IsMap() && ModuleConfig["name"]) {
                Message += " [" + ModuleConfig["name"].as<std::string>() + "]";
            }
            IO::GlobalLogger << IO::LI_erro() + Message + "\n";
            IO::GlobalLogger << IO::LI_erro() + e.what() + "\n";
            throw std::runtime_error(Message + ": " + e.what());
        } catch (const std::exception& e) {
            IO::GlobalLogger << IO::LI_erro() + e.what() + "\n";
            throw;
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
