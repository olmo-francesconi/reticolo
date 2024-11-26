/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: factory/AlgorithmBase.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

*******************************************************************************/

#pragma once

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <format>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include "reticolo/action/RelativisticBoseGas.hpp"
#include "reticolo/action/WeakFieldEuclideanGR.hpp"
#include "reticolo/core/types/real.hpp"
#include "reticolo/modules/factory/ModuleBase.hpp"
#include "reticolo/modules/montecarlo/MonteCarloHandler.hpp"

namespace reticolo {

enum class Modules : std::uint8_t {
    MonteCarlo,
};
std::map<std::string, Modules> ModMap{{"MonteCarlo", Modules::MonteCarlo}};

enum class Actions : std::uint8_t {
    RelativisticBoseGasF,
    RelativisticBoseGasD,
    WeakFieldEuclideanGRF,
    WeakFieldEuclideanGRD,
};
std::map<std::string, Actions> ActMap{{"RelativisticBoseGas", Actions::RelativisticBoseGasD},      //
                                      {"RelativisticBoseGas_F", Actions::RelativisticBoseGasF},    //
                                      {"RelativisticBoseGas_D", Actions::RelativisticBoseGasD},    //
                                      {"WeakFieldEuclideanGR", Actions::WeakFieldEuclideanGRD},    //
                                      {"WeakFieldEuclideanGR_F", Actions::WeakFieldEuclideanGRF},  //
                                      {"WeakFieldEuclideanGR_D", Actions::WeakFieldEuclideanGRD}};

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
                        return std::make_unique<MMonteCarlo::MonteCarloHandler<action::RelativisticBoseGas<RealF>>>();
                    case Actions::RelativisticBoseGasD:
                        return std::make_unique<MMonteCarlo::MonteCarloHandler<action::RelativisticBoseGas<RealD>>>();
                    case Actions::WeakFieldEuclideanGRF:
                        return std::make_unique<MMonteCarlo::MonteCarloHandler<action::WeakFieldEuclideanGR<RealF>>>();
                    case Actions::WeakFieldEuclideanGRD:
                        return std::make_unique<MMonteCarlo::MonteCarloHandler<action::WeakFieldEuclideanGR<RealD>>>();
                    default:
                        throw std::runtime_error("Action not valid for selected module");
                        return nullptr;
                }
            default:
                throw std::runtime_error("Module not found");
                return nullptr;
        }
    }
};

}  // namespace reticolo
