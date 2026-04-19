/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/registration/RelativisticBoseGasModuleRegistration.hpp

*******************************************************************************/

#pragma once

#include "reticolo/action/RelativisticBoseGas.hpp"
#include "reticolo/action/registration/ActionModuleRegistrationSupport.hpp"

namespace reticolo::registration {

inline void register_relativistic_bose_gas_modules() {
    register_monte_carlo_action_family<RelativisticBoseGasDescriptor, action::RelativisticBoseGas>();
}

}  // namespace reticolo::registration
