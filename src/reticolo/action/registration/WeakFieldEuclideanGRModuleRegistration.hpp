/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/registration/WeakFieldEuclideanGRModuleRegistration.hpp

*******************************************************************************/

#pragma once

#include "reticolo/action/WeakFieldEuclideanGR.hpp"
#include "reticolo/action/registration/ActionModuleRegistrationSupport.hpp"

namespace reticolo::registration {

inline void register_weak_field_euclidean_gr_modules() {
    register_monte_carlo_action_family<WeakFieldEuclideanGRDescriptor, action::WeakFieldEuclideanGR>();
}

}  // namespace reticolo::registration
