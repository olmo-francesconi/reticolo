/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: factory/BuiltinModuleRegistration.hpp

*******************************************************************************/

#pragma once

#include "reticolo/action/registration/RelativisticBoseGasModuleRegistration.hpp"
#include "reticolo/action/registration/TestActionModuleRegistration.hpp"
#include "reticolo/action/registration/WeakFieldEuclideanGRModuleRegistration.hpp"

namespace reticolo::registration {

inline void register_builtin_modules() {
#define RETICOLO_BUILTIN_ACTION_FAMILY(DESCRIPTOR, REGISTER_FN) REGISTER_FN();
#include "reticolo/action/registration/BuiltinActionFamilies.def"
#undef RETICOLO_BUILTIN_ACTION_FAMILY
}

}  // namespace reticolo::registration
