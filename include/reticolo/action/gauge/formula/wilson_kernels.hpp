#pragma once

// Per-gauge-group Wilson plaquette kernels — the action-specific physics of the
// Wilson action, kept OUT of the gauge-group models (which hold only the core
// group operations: constants + the HMC algebra sample/kinetic/expi hooks).
//
// Each group specializes `wilson_kernels<G>` in action/gauge/formula/wilson_<g>.hpp
// with the plaquette Re Tr, the Σ Re Tr plane fast-path, and the staple force /
// fused-kick scatter. `Wilson<G>` (action/gauge/wilson.hpp) dispatches to it.

namespace reticolo::action::formula {

template <class G>
struct wilson_kernels;  // primary left undefined; each group provides a specialization

}  // namespace reticolo::action::formula
