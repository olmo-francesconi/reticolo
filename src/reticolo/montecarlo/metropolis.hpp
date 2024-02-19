/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: montecarlo/metropolis.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include "H5Cpp.h"
#include "reticolo/lattice/lattice.hpp"
#include "reticolo/montecarlo/montecarlo_data.hpp"
#include "reticolo/tools/types/basic.hpp"

namespace reticolo::montecarlo {

template <class Action>
class Metropolis_worker {
 private:
  montecarlo::data<typename Action::ActionType> MC_data;
  void sweep(lattice<typename Action::FieldType, Action::dims>& field);

 public:
  Metropolis_worker() = default;

  void init();

  void run(uint steps);
};

}  // namespace reticolo::montecarlo
