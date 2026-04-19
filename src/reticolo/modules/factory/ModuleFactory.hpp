/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: factory/ModuleFactory.hpp

*******************************************************************************/

#pragma once

#include <format>
#include <memory>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

#include "reticolo/modules/factory/BuiltinModuleRegistration.hpp"
#include "reticolo/modules/factory/ModuleRegistry.hpp"
#include "reticolo/runtime/BuiltinMetadata.hpp"

namespace reticolo {

class ModuleFactory {
  public:
    static void EnsureBuiltinsRegistered() { registration::register_builtin_modules(); }

    [[nodiscard]] static auto AvailableActions(const std::string& module_name) -> std::vector<std::string> {
        EnsureBuiltinsRegistered();
        std::vector<std::string> actions;
        for (const auto action : runtime::metadata::actions_for_module(module_name)) {
            actions.emplace_back(action);
        }
        return actions;
    }

    static void ValidateModuleAction(const std::string& module_name, const std::string& action_name) {
        EnsureBuiltinsRegistered();
        if (ModuleRegistry::instance().contains(module_name, action_name)) {
            return;
        }

        const auto Family = runtime::metadata::describe_action(action_name);
        if (Family.has_value() && Family->module_name == module_name) {
            throw std::runtime_error(
                std::format("Action '{}' belongs to module '{}', but no concrete implementation is registered",
                            action_name, module_name));
        }

        const auto Available = AvailableActions(module_name);
        if (Available.empty()) {
            throw std::runtime_error(std::format("Requested module '{}' is not registered", module_name));
        }

        throw std::runtime_error(
            std::format("Requested action '{}' is not registered for module '{}'. Available actions: {}", action_name,
                        module_name, fmt_join(Available)));
    }

    static auto MakeModule(const std::string& module_name, const std::string& action_name)
        -> std::unique_ptr<ModuleBase> {
        ValidateModuleAction(module_name, action_name);
        return ModuleRegistry::instance().create(module_name, action_name);
    }

  private:
    [[nodiscard]] static auto fmt_join(const std::vector<std::string>& values) -> std::string {
        std::string out;
        for (std::size_t i = 0; i < values.size(); i++) {
            if (i != 0) {
                out += ", ";
            }
            out += values[i];
        }
        return out;
    }
};

}  // namespace reticolo
