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

#include "reticolo/modules/factory/MCAlgorithmBase.hpp"
#include "reticolo/modules/montecarlo/MonteCarloData.hpp"
#include "reticolo/modules/montecarlo/algorithms/HMC.hpp"
#include "reticolo/modules/montecarlo/algorithms/Metropolis.hpp"

namespace reticolo::MMonteCarlo {

enum class MCAlgorithms {
    Metropolis,
    HMC,
};

inline std::map<std::string, MCAlgorithms> AlgMap{
    {"Metropolis", MCAlgorithms::Metropolis},
    {"HMC", MCAlgorithms::HMC},
};

template <class Action>
class MCAlgorithmFactory {
    using action_type = Action::ActionType;
    using field_type = typename Action::FieldType;
    using observables_type = typename Action::Observables;
    using monte_carlo_data_type = MMonteCarlo::data<typename Action::ActionType>;

  public:
    static auto MakeUpdater(const std::string& name) -> std::unique_ptr<MCAlgorithmBase<Action>> {
        auto Alg = AlgMap.find(name);

        switch (Alg->second) {
            case MCAlgorithms::Metropolis:
                return std::make_unique<Metropolis<Action>>();
            case MCAlgorithms::HMC:
                return std::make_unique<HMC<Action>>();
            default:
                return nullptr;
        }
    }
};

}  // namespace reticolo::MMonteCarlo
