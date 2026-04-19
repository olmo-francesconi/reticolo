/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: ReticoloCore.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 - This file defines the ReticoloCore class, which implements the Singleton
   design pattern. It provides mechanisms for initializing a single instance,
   loading configurations from a YAML file, and accessing the setup data.

*******************************************************************************/

#pragma once

#include <filesystem>
#include <stdexcept>
#include <string>

#include "yaml-cpp/node/node.h"
#include "yaml-cpp/node/parse.h"
#include "yaml-cpp/yaml.h"  // IWYU pragma: keep

namespace reticolo {

class ReticoloCore {
  protected:
    ReticoloCore() = default;
    inline static ReticoloCore* Instance = nullptr;
    inline static YAML::Node    Setup = YAML::Node("");

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
        (void)std::filesystem::canonical(filename);
        Setup = YAML::LoadFile(filename);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to load configuration '" + filename + "': " + e.what());
    }
};

auto ReticoloCore::getSetup() -> const YAML::Node& { return Setup; };

}  // namespace reticolo
