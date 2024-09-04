/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: factory/AlgorithmBase.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <string>

#include "reticolo/action/RelativisticBoseGas.hpp"
#include "reticolo/modules/factory/ModuleBase.hpp"
#include "reticolo/modules/montecarlo/MonteCarloHandler.hpp"

namespace reticolo {

enum class Modules {
    Montecarlo,
};
std::map<std::string, Modules> ModMap{{"MonteCarlo", Modules::Montecarlo}};

enum class Actions {
    RelativisticBoseGas,
    WeakFieldEuclideanGR,
};
std::map<std::string, Actions> ActMap{{"RelativisticBoseGas", Actions::RelativisticBoseGas},
                                      {"WeakFieldEuclideanGR", Actions::WeakFieldEuclideanGR}};

class ModuleFactory {
  public:
    static auto MakeModule(const std::string& name, const std::string& action) -> std::unique_ptr<ModuleBase> {
        auto Mod = ModMap.find(name);
        auto Act = ActMap.find(action);

        switch (Mod->second) {
            case Modules::Montecarlo:
                switch (Act->second) {
                    case Actions::RelativisticBoseGas:
                        return std::make_unique<montecarlo::MonteCarloHandler<action::RelativisticBoseGas>>();
                    default:
                        return nullptr;
                }
            default:
                return nullptr;
        }
    }
};

}  // namespace reticolo
