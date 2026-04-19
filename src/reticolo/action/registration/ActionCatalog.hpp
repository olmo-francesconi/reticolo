/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/registration/ActionCatalog.hpp

*******************************************************************************/

#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <string_view>
#include <vector>

#include "reticolo/action/registration/ActionDescriptor.hpp"

namespace reticolo::registration {

constexpr std::size_t builtin_action_family_count() {
    std::size_t Count = 0;
#define RETICOLO_BUILTIN_ACTION_FAMILY(DESCRIPTOR, REGISTER_FN) ++Count;
#include "reticolo/action/registration/BuiltinActionFamilies.def"
#undef RETICOLO_BUILTIN_ACTION_FAMILY
    return Count;
}

[[nodiscard]] inline auto builtin_action_catalog()
    -> const std::array<ActionManifestView, builtin_action_family_count()>& {
    static constexpr auto Catalog = std::array<ActionManifestView, builtin_action_family_count()>{
#define RETICOLO_BUILTIN_ACTION_FAMILY(DESCRIPTOR, REGISTER_FN) descriptor_manifest<DESCRIPTOR>(),
#include "reticolo/action/registration/BuiltinActionFamilies.def"
#undef RETICOLO_BUILTIN_ACTION_FAMILY
    };
    return Catalog;
}

[[nodiscard]] inline auto available_modules() -> std::vector<std::string_view> {
    std::vector<std::string_view> modules;
    for (const auto& family : builtin_action_catalog()) {
        if (std::find(modules.begin(), modules.end(), family.module_name) == modules.end()) {
            modules.push_back(family.module_name);
        }
    }
    return modules;
}

[[nodiscard]] inline auto actions_for_module(std::string_view module_name) -> std::vector<std::string_view> {
    std::vector<std::string_view> actions;
    for (const auto& family : builtin_action_catalog()) {
        if (family.module_name != module_name) {
            continue;
        }
        actions.push_back(family.default_name);
        if (family.has_float_precision) {
            actions.push_back(family.float_name);
        }
        if (family.has_double_precision) {
            actions.push_back(family.double_name);
        }
        actions.insert(actions.end(), family.aliases.begin(), family.aliases.end());
    }
    return actions;
}

[[nodiscard]] inline auto find_action_family(std::string_view module_name, std::string_view action_name)
    -> const ActionManifestView* {
    for (const auto& family : builtin_action_catalog()) {
        if (family.module_name != module_name) {
            continue;
        }
        if (family.matches_action_name(action_name)) {
            return std::addressof(family);
        }
    }
    return nullptr;
}

[[nodiscard]] inline auto find_action_family(std::string_view action_name) -> const ActionManifestView* {
    for (const auto& family : builtin_action_catalog()) {
        if (family.matches_action_name(action_name)) {
            return std::addressof(family);
        }
    }
    return nullptr;
}

[[nodiscard]] inline auto algorithms_for_action(std::string_view action_name) -> std::vector<std::string_view> {
    const auto* Family = find_action_family(action_name);
    if (Family == nullptr) {
        return {};
    }
    return std::vector<std::string_view>(Family->algorithms.begin(), Family->algorithms.end());
}

}  // namespace reticolo::registration
