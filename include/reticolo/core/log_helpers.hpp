#pragma once

// Domain-aware logging helpers. Sits on top of <reticolo/core/log.hpp> so
// the core logger stays domain-agnostic — anything that knows about
// actions, algorithms, integrators, observables, etc. lives here.

#include <reticolo/core/log.hpp>

namespace reticolo::log {

// Announce an action — uniform header line for any action that exposes a
// `void describe(log::Entry&) const` method. The action picks its content
// (concept name + parameters via .line() and .param()); the helper picks
// the tag and level so output is consistent across all actions.
template <class Action>
void act(Action const& a) {
    Entry e{Level::info, "act"};
    a.describe(e);
}

// Announce an algorithm — same shape as `act`, but the tag comes from the
// algorithm class itself (`A::log_tag`) since each algorithm gets its own
// 4-char tag (`hmc`, `metr`, `wolf`, …) for the per-step lines.
template <class A>
void algo(A const& a) {
    Entry e{Level::info, A::log_tag};
    a.describe(e);
}

}  // namespace reticolo::log
