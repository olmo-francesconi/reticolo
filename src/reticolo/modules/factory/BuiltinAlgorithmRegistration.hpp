/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: factory/BuiltinAlgorithmRegistration.hpp

*******************************************************************************/

#pragma once

#include <random>

#include "reticolo/action/adapters/BuiltinMonteCarloAdapters.hpp"
#include "reticolo/action/registration/ActionDescriptor.hpp"
#include "reticolo/modules/montecarlo/algorithms/HMC.hpp"
#include "reticolo/modules/montecarlo/algorithms/Metropolis.hpp"

namespace reticolo::MMonteCarlo::AlgorithmFactory {

template <class Action, class TGen = std::mt19937_64>
inline void register_builtin_algorithms() {
    using descriptor = ::reticolo::registration::action_descriptor_t<Action>;

    if constexpr (descriptor::supports_metropolis) {
        register_metropolis_algorithm<Action, TGen>();
    }

    if constexpr (descriptor::supports_hmc) {
        register_hmc_algorithm<Action, TGen>();
    }

    if constexpr (descriptor::supports_llr) {
        register_llr_metropolis_algorithm<Action, TGen>();
    }
}

}  // namespace reticolo::MMonteCarlo::AlgorithmFactory
