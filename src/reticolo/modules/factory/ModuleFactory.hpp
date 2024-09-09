/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: factory/AlgorithmBase.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

*******************************************************************************/

#pragma once

#include <cstddef>
#include <format>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include "reticolo/action/RelativisticBoseGas.hpp"
#include "reticolo/modules/factory/ModuleBase.hpp"
#include "reticolo/modules/montecarlo/MonteCarloHandler.hpp"

namespace reticolo {

enum class Modules {
    MonteCarlo,
};
std::map<std::string, Modules> ModMap{{"MonteCarlo", Modules::MonteCarlo}};

enum class Actions {
    RelativisticBoseGasF,
    RelativisticBoseGasD,
    WeakFieldEuclideanGR,
};
std::map<std::string, Actions> ActMap{{"RelativisticBoseGasF", Actions::RelativisticBoseGasF},
                                      {"RelativisticBoseGasD", Actions::RelativisticBoseGasD},
                                      {"WeakFieldEuclideanGR", Actions::WeakFieldEuclideanGR}};

class ModuleFactory {
  public:
    static auto MakeModule(const std::string& module_name, const std::string& action) -> std::unique_ptr<ModuleBase> {
        auto Mod = ModMap.find(module_name);
        auto Act = ActMap.find(action);

        if (Mod == ModMap.end()) {
            throw std::runtime_error(std::format("Requested Module ({}) not found", action));
            return nullptr;
        }

        if (Act == ActMap.end()) {
            throw std::runtime_error(std::format("Requested Action ({}) not found", action));
            return nullptr;
        }

        switch (Mod->second) {
            case Modules::MonteCarlo:
                switch (Act->second) {
                    case Actions::RelativisticBoseGasF:
                        return std::make_unique<MMonteCarlo::MonteCarloHandler<action::RelativisticBoseGasF>>();
                    case Actions::RelativisticBoseGasD:
                        return std::make_unique<MMonteCarlo::MonteCarloHandler<action::RelativisticBoseGasD>>();
                    default:
                        return nullptr;
                }
            default:
                return nullptr;
        }
    }
};

}  // namespace reticolo
