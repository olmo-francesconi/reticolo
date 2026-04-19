/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: modules/llr/LegacyLLRSupport.hpp

*******************************************************************************/

#pragma once

#include "reticolo/action/registration/ActionDescriptor.hpp"

namespace reticolo::legacy::llr {

// The current LLR subsystem is a legacy path: it is not yet wired into the
// registry-driven module/runtime metadata flow used by the built-in runtime.
// Keep descriptor coupling centralized here so future normalization or removal
// can happen in one place.

template <class Action>
concept llr_capable_action = ::reticolo::registration::action_descriptor_t<Action>::supports_llr;

inline constexpr bool is_registry_integrated = false;

}  // namespace reticolo::legacy::llr
