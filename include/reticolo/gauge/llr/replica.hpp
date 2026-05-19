#pragma once

// Transitional shim: gauge LLR `Replica` is the unified scalar template
// instantiated on `LinkLattice<T>`. To be deleted once apps switch to the
// unqualified `reticolo::llr::Replica` name.

#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/link_lattice.hpp>
#include <reticolo/gauge/llr/windowed_action.hpp>
#include <reticolo/llr/replica.hpp>

namespace reticolo::gauge::llr {

template <class Base,
          class Rng,
          class Integrator = reticolo::alg::integ::Leapfrog,
          class T          = typename Base::value_type>
using Replica = reticolo::llr::Replica<Base, Rng, Integrator, T, LinkLattice<T>>;

}  // namespace reticolo::gauge::llr
