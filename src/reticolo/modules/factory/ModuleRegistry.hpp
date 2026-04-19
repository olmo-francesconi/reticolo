/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: factory/ModuleRegistry.hpp

*******************************************************************************/

#pragma once

#include <format>
#include <functional>
#include <memory>
#include <ranges>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "reticolo/modules/factory/ModuleBase.hpp"

namespace reticolo {

class ModuleRegistry {
  public:
    using creator_type = std::function<std::unique_ptr<ModuleBase>()>;

    static auto instance() -> ModuleRegistry& {
        static ModuleRegistry Registry;
        return Registry;
    }

    void register_module(const std::string& module_name, const std::string& action_name, creator_type creator) {
        _Creators[make_key(module_name, action_name)] = std::move(creator);
    }

    [[nodiscard]] auto contains(const std::string& module_name, const std::string& action_name) const -> bool {
        return _Creators.contains(make_key(module_name, action_name));
    }

    [[nodiscard]] auto actions_for_module(const std::string& module_name) const -> std::vector<std::string> {
        std::vector<std::string> actions;
        const auto               prefix = module_name + "::";
        for (const auto& [key, _] : _Creators) {
            if (key.rfind(prefix, 0) == 0) {
                actions.push_back(key.substr(prefix.size()));
            }
        }
        std::ranges::sort(actions);
        actions.erase(std::unique(actions.begin(), actions.end()), actions.end());
        return actions;
    }

    auto create(const std::string& module_name, const std::string& action_name) const -> std::unique_ptr<ModuleBase> {
        const auto It = _Creators.find(make_key(module_name, action_name));
        if (It == _Creators.end()) {
            throw std::runtime_error(
                std::format("Requested module/action combination not registered: module='{}', action='{}'", module_name,
                            action_name));
        }
        return (It->second)();
    }

  private:
    [[nodiscard]] static auto make_key(const std::string& module_name, const std::string& action_name) -> std::string {
        return module_name + "::" + action_name;
    }

    std::unordered_map<std::string, creator_type> _Creators;
};

template <typename ModuleT>
inline void register_module_type(const std::string& module_name, const std::string& action_name) {
    ModuleRegistry::instance().register_module(module_name, action_name, []() { return std::make_unique<ModuleT>(); });
}

}  // namespace reticolo
