/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/registration/ActionModuleRegistrationSupport.hpp

*******************************************************************************/

#pragma once

#include "reticolo/action/registration/ActionDescriptor.hpp"
#include "reticolo/core/types/real.hpp"
#include "reticolo/modules/factory/ModuleRegistry.hpp"
#include "reticolo/modules/montecarlo/MonteCarloHandler.hpp"

namespace reticolo::registration {

template <template <typename> class ActionTemplate>
struct PrecisionModuleBinding {
    using float_action_type = ActionTemplate<RealF>;
    using double_action_type = ActionTemplate<RealD>;
    using float_module_type = MMonteCarlo::MonteCarloHandler<float_action_type>;
    using double_module_type = MMonteCarlo::MonteCarloHandler<double_action_type>;
};

template <typename Binding>
inline void register_precision_bound_module(ActionPrecisionBinding precision, const std::string& module_name,
                                            const std::string& action_name) {
    switch (precision) {
        case ActionPrecisionBinding::float_precision:
            register_module_type<typename Binding::float_module_type>(module_name, action_name);
            break;
        case ActionPrecisionBinding::double_precision:
            register_module_type<typename Binding::double_module_type>(module_name, action_name);
            break;
    }
}

template <typename Descriptor, template <typename> class ActionTemplate>
inline void register_monte_carlo_action_family() {
    using binding = PrecisionModuleBinding<ActionTemplate>;
    constexpr auto Manifest = descriptor_manifest<Descriptor>();

    static const bool Registered = [Manifest]() {
        register_precision_bound_module<binding>(Manifest.canonical_precision, std::string(Manifest.module_name),
                                                 std::string(Manifest.default_name));

        if constexpr (Descriptor::has_float_precision) {
            register_module_type<typename binding::float_module_type>(std::string(Manifest.module_name),
                                                                      std::string(Manifest.float_name));
        }
        if constexpr (Descriptor::has_double_precision) {
            register_module_type<typename binding::double_module_type>(std::string(Manifest.module_name),
                                                                       std::string(Manifest.double_name));
        }

        for (const auto alias : Manifest.aliases) {
            register_precision_bound_module<binding>(Manifest.alias_precision, std::string(Manifest.module_name),
                                                     std::string(alias));
        }
        return true;
    }();
    (void)Registered;
}

}  // namespace reticolo::registration
