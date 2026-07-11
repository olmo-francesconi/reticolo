#pragma once

// Aggregator for the nearest-neighbour scalar action family — fields on a
// `Lattice<T>` whose energy is one traversal folding a per-bond COMBINE over the
// neighbours and a per-site FINALIZE. Covers both the identity-combine "self +
// neighbour-sum" actions (Phi4 / Phi6 / SineGordon) and the endpoint-difference
// bond actions (XY). Each leaf derives from `NNAction` and carries only its
// formula (in `nn/formula/`); see `action/concepts.hpp` for the HMC contract.

// NOLINTBEGIN(misc-include-cleaner): re-exports are the point of the aggregator.
#include <reticolo/action/nn/phi4.hpp>
#include <reticolo/action/nn/phi6.hpp>
#include <reticolo/action/nn/sine_gordon.hpp>
#include <reticolo/action/nn/xy.hpp>
// NOLINTEND(misc-include-cleaner)
