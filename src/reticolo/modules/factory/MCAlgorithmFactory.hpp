/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: factory/AlgorithmBase.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

*******************************************************************************/

#pragma once

#include <cstddef>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>

#include "reticolo/modules/factory/MCAlgorithmBase.hpp"
#include "reticolo/modules/montecarlo/algorithms/HMC.hpp"
#include "reticolo/modules/montecarlo/algorithms/Metropolis.hpp"

namespace reticolo::MMonteCarlo {

// template <class Action>
// class MCAlgorithmFactory {
//   public:
//     static auto MakeUpdater(const std::string& name) -> std::unique_ptr<MCAlgorithmBase<Action>> {
//         if (name == "Metropolis") {
//             return std::make_unique<Metropolis<Action>>();
//         }

//         if (name == "HMC") {
//             return std::make_unique<HMC<Action>>();
//         }

//         throw std::runtime_error(std::format("Requested MonteCarlo update algorithm ({}) not implemented", name));
//         return nullptr;
//     }
// };

namespace AlgorithmFactory {
template <class Action>
static auto MakeUpdater(const std::string& name) -> std::unique_ptr<MCAlgorithmBase<Action>> {
    if (name == "Metropolis") {
        return std::make_unique<Metropolis<Action>>();
    }

    if (name == "HMC") {
        return std::make_unique<HMC<Action>>();
    }

    throw std::runtime_error(std::format("Requested MonteCarlo update algorithm ({}) not implemented", name));
    return nullptr;
}
}  // namespace AlgorithmFactory

}  // namespace reticolo::MMonteCarlo
