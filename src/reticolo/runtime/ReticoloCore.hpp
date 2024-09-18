/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: ReticoloCore.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

*******************************************************************************/

#pragma once

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#include "yaml-cpp/node/node.h"
#include "yaml-cpp/node/parse.h"
#include "yaml-cpp/yaml.h"  // IWYU pragma: keep

namespace reticolo {

class ReticoloCore {
  protected:
    ReticoloCore() = default;
    static ReticoloCore* Instance;
    static YAML::Node    Setup;

  public:
    /* Singletons should not be cloneable and assignable */
    ReticoloCore(ReticoloCore& other) = delete;
    void operator=(const ReticoloCore&) = delete;

    /* Create and get access to the singleton */
    static auto Init() -> ReticoloCore*;

    /* Load setting from file */
    static void readSetup(const std::string& filename);

    /* Broadcast setup */
    static auto getSetup() -> const YAML::Node&;

    // [[nodiscard]] auto Id() const -> std::string { return _Id; }
};

/*--------------------------------------------------------------------------------------------------
  Initialize static memeber variables
--------------------------------------------------------------------------------------------------*/
/* Instance initialized as null pointer */
ReticoloCore* ReticoloCore::Instance = nullptr;

/* Initialize the Setup file as an empty file */
YAML::Node ReticoloCore::Setup = YAML::Node("");

/*--------------------------------------------------------------------------------------------------
  Static methods definition
--------------------------------------------------------------------------------------------------*/
auto ReticoloCore::Init() -> ReticoloCore* {
    if (Instance == nullptr) {
        Instance = new ReticoloCore();
    }
    return Instance;
}

void ReticoloCore::readSetup(const std::string& filename) {
    try {
        auto ConfigFilePath = std::filesystem::canonical(filename);
        Setup = YAML::LoadFile(filename);
    } catch (const std::exception& e) {
        std::cout << "parse config file failed:\n" << e.what() << std::endl;
        exit(EXIT_FAILURE);
    }
};

auto ReticoloCore::getSetup() -> const YAML::Node& { return Setup; };

}  // namespace reticolo
