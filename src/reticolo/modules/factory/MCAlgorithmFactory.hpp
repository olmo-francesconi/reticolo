/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: factory/AlgorithmBase.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

*******************************************************************************/

#pragma once

#include <cstddef>
#include <format>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>

#include "reticolo/modules/factory/MCAlgorithmBase.hpp"
#include "reticolo/modules/montecarlo/algorithms/HMC.hpp"
#include "reticolo/modules/montecarlo/algorithms/Metropolis.hpp"

namespace reticolo::MMonteCarlo::AlgorithmFactory {

template <class Action, class TGen = std::mt19937_64>
static auto MakeUpdater(const std::string& name) -> std::unique_ptr<MCAlgorithmBase<Action, TGen>> {
    if constexpr (Action::IsMetropolisCapable) {
        if ((name == "Metropolis")) {
            return std::make_unique<Metropolis<Action>>();
        }
    }

    if constexpr (Action::IsHmcCapable) {
        if ((name == "HMC")) {
            return std::make_unique<HMC<Action>>();
        }
    }

    if constexpr (Action::IsLLRCapable) {
        if ((name == "LLRMetropolis")) {
            return std::make_unique<HMC<Action>>();
        }
    }

    throw std::runtime_error(std::format("Requested MonteCarlo update algorithm ({}) not implemented", name));
    return nullptr;
}
}  // namespace reticolo::MMonteCarlo::AlgorithmFactory
