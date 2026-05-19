#pragma once

// Transitional shim: gauge LLR `WindowedAction` is the unified scalar
// template instantiated on `LinkLattice<T>`. To be deleted once apps switch
// to the unqualified `reticolo::llr::WindowedAction` name.

#include <reticolo/core/link_lattice.hpp>
#include <reticolo/llr/windowed_action.hpp>

namespace reticolo::gauge::llr {

template <class Base, class T = typename Base::value_type>
using WindowedAction = reticolo::llr::WindowedAction<Base, T, LinkLattice<T>>;

}  // namespace reticolo::gauge::llr
