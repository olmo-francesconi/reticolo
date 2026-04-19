#pragma once

#include "reticolo/action/TestAction.hpp"
#include "reticolo/action/registration/ActionModuleRegistrationSupport.hpp"

namespace reticolo::registration {

inline void register_test_action_modules() {
    register_monte_carlo_action_family<TestActionDescriptor, action::TestAction>();
}

}  // namespace reticolo::registration
