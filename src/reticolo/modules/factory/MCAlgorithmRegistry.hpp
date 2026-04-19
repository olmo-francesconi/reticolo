/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: factory/MCAlgorithmRegistry.hpp

*******************************************************************************/

#pragma once

#include <format>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <unordered_map>

#include "reticolo/modules/factory/MCAlgorithmBase.hpp"

namespace reticolo::MMonteCarlo {

template <class Action, class TGen = std::mt19937_64>
class MCAlgorithmRegistry {
  public:
    using creator_type = std::function<std::unique_ptr<MCAlgorithmBase<Action, TGen>>()>;

    static auto instance() -> MCAlgorithmRegistry& {
        static MCAlgorithmRegistry Registry;
        return Registry;
    }

    void register_algorithm(const std::string& name, creator_type creator) { _Creators[name] = std::move(creator); }

    [[nodiscard]] auto contains(const std::string& name) const -> bool { return _Creators.contains(name); }

    auto create(const std::string& name) const -> std::unique_ptr<MCAlgorithmBase<Action, TGen>> {
        const auto It = _Creators.find(name);
        if (It == _Creators.end()) {
            throw std::runtime_error(std::format("Requested MonteCarlo update algorithm ({}) not implemented", name));
        }
        return (It->second)();
    }

  private:
    std::unordered_map<std::string, creator_type> _Creators;
};

template <class Action, class AlgorithmT, class TGen = std::mt19937_64>
inline void register_algorithm_type(const std::string& name) {
    MCAlgorithmRegistry<Action, TGen>::instance().register_algorithm(name,
                                                                     []() { return std::make_unique<AlgorithmT>(); });
}

}  // namespace reticolo::MMonteCarlo
