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

#include "reticolo/modules/montecarlo/MonteCarloData.hpp"
#include "reticolo/modules/montecarlo/algorithms/AlgorithmBase.hpp"
#include "reticolo/modules/montecarlo/algorithms/HMC.hpp"
#include "reticolo/modules/montecarlo/algorithms/LLRHMC.hpp"
#include "reticolo/modules/montecarlo/algorithms/LLRHMCMetropolis.hpp"
#include "reticolo/modules/montecarlo/algorithms/LLRMetropolis.hpp"
#include "reticolo/modules/montecarlo/algorithms/Metropolis.hpp"

namespace reticolo::montecarlo {

enum class Algorithms {
    Metropolis,
    HMC,
    LLRMetropolis,
    LLRHMC,
    LLRHMCMetropolis,
};

std::map<std::string, Algorithms> AlgMap{{"Metropolis", Algorithms::Metropolis},
                                         {"HMC", Algorithms::HMC},
                                         {"LLRMetropolis", Algorithms::LLRMetropolis},
                                         {"LLRHMC", Algorithms::LLRHMC},
                                         {"LLRHMCMetropolis", Algorithms::LLRHMCMetropolis}};

template <class Action>
class AlgorithmFactory {
    using action_type = Action::ActionType;
    using field_type = typename Action::FieldType;
    using observables_type = typename Action::Observables;
    using monte_carlo_data_type = montecarlo::data<typename Action::ActionType>;

  public:
    static auto MakeUpdater(const std::string& name) -> std::unique_ptr<AlgorithmBase<Action>> {
        auto Alg = AlgMap.find(name);

        switch (Alg->second) {
            case Algorithms::Metropolis:
                return std::make_unique<Metropolis<Action>>();
            case Algorithms::HMC:
                return std::make_unique<HMC<Action>>();
            case Algorithms::LLRMetropolis:
                return std::make_unique<LLRMetropolis<Action>>();
            case Algorithms::LLRHMC:
                return std::make_unique<LLRHMC<Action>>();
            case Algorithms::LLRHMCMetropolis:
                return std::make_unique<LLRHMCMetropolis<Action>>();
            default:
                return nullptr;
        }
    }
};

}  // namespace reticolo::montecarlo
