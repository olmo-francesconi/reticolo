/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: runtime/BuiltinMetadata.hpp

*******************************************************************************/

#pragma once

#include <optional>
#include <string_view>
#include <vector>

#include "reticolo/action/registration/ActionCatalog.hpp"

namespace reticolo::runtime::metadata {

struct ActionInfo {
    std::string_view                     module_name;
    std::string_view                     canonical_name;
    std::optional<std::string_view>      float_name;
    std::optional<std::string_view>      double_name;
    std::vector<std::string_view>        aliases;
    std::vector<std::string_view>        algorithms;
    bool                                 supports_metropolis;
    bool                                 supports_hmc;
    bool                                 supports_llr;
    registration::ActionPrecisionBinding canonical_precision;
    registration::ActionPrecisionBinding alias_precision;
    std::string_view                     storage_schema;
};

[[nodiscard]] inline auto available_modules() -> std::vector<std::string_view> {
    return registration::available_modules();
}

[[nodiscard]] inline auto actions_for_module(std::string_view module_name) -> std::vector<std::string_view> {
    return registration::actions_for_module(module_name);
}

[[nodiscard]] inline auto describe_action(std::string_view action_name) -> std::optional<ActionInfo> {
    const auto* Family = registration::find_action_family(action_name);
    if (Family == nullptr) {
        return std::nullopt;
    }

    return ActionInfo{
        .module_name = Family->module_name,
        .canonical_name = Family->default_name,
        .float_name = Family->has_float_precision ? std::optional<std::string_view>(Family->float_name) : std::nullopt,
        .double_name =
            Family->has_double_precision ? std::optional<std::string_view>(Family->double_name) : std::nullopt,
        .aliases = std::vector<std::string_view>(Family->aliases.begin(), Family->aliases.end()),
        .algorithms = std::vector<std::string_view>(Family->algorithms.begin(), Family->algorithms.end()),
        .supports_metropolis = Family->supports_metropolis,
        .supports_hmc = Family->supports_hmc,
        .supports_llr = Family->supports_llr,
        .canonical_precision = Family->canonical_precision,
        .alias_precision = Family->alias_precision,
        .storage_schema = Family->storage_schema,
    };
}

[[nodiscard]] inline auto canonical_action_name(std::string_view action_name) -> std::optional<std::string_view> {
    const auto Info = describe_action(action_name);
    if (!Info.has_value()) {
        return std::nullopt;
    }
    return Info->canonical_name;
}

[[nodiscard]] inline auto algorithms_for_action(std::string_view action_name) -> std::vector<std::string_view> {
    const auto Info = describe_action(action_name);
    if (!Info.has_value()) {
        return {};
    }
    return Info->algorithms;
}

}  // namespace reticolo::runtime::metadata
