/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: factory/MCAlgorithmFactory.hpp

*******************************************************************************/

#pragma once

#include <format>
#include <memory>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "reticolo/action/registration/ActionDescriptor.hpp"
#include "reticolo/modules/factory/BuiltinAlgorithmRegistration.hpp"
#include "reticolo/modules/factory/MCAlgorithmRegistry.hpp"
#include "reticolo/runtime/BuiltinMetadata.hpp"

namespace reticolo::MMonteCarlo::AlgorithmFactory {

template <typename StringLike>
[[nodiscard]] static auto fmt_join(const std::vector<StringLike>& values) -> std::string {
    std::string out;
    for (std::size_t i = 0; i < values.size(); i++) {
        if (i != 0) {
            out += ", ";
        }
        out += values[i];
    }
    return out;
}

template <class Action, class TGen = std::mt19937_64>
static void ValidateUpdaterName(const std::string& name) {
    register_builtin_algorithms<Action, TGen>();
    if (MCAlgorithmRegistry<Action, TGen>::instance().contains(name)) {
        return;
    }

    const auto Available = ::reticolo::runtime::metadata::algorithms_for_action(
        ::reticolo::registration::action_descriptor_t<Action>::default_name);
    throw std::runtime_error(
        std::format("Requested MonteCarlo update algorithm '{}' not available for action '{}'. "
                    "Available algorithms: {}",
                    name, ::reticolo::registration::action_descriptor_t<Action>::default_name, fmt_join(Available)));
}

template <class Action, class TGen = std::mt19937_64>
static auto MakeUpdater(const std::string& name) -> std::unique_ptr<MCAlgorithmBase<Action, TGen>> {
    ValidateUpdaterName<Action, TGen>(name);
    return MCAlgorithmRegistry<Action, TGen>::instance().create(name);
}

}  // namespace reticolo::MMonteCarlo::AlgorithmFactory
