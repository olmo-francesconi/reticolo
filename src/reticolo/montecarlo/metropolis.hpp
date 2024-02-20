/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: montecarlo/metropolis.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include "H5Cpp.h"
#include "reticolo/lattice/lattice.hpp"
#include "reticolo/montecarlo/montecarlo_data.hpp"
#include "reticolo/types/core.hpp"

namespace reticolo::montecarlo {

template <class Action>
class MetropolisWorker {
  private:
    montecarlo::data<typename Action::ActionType> _McData;

    void sweep(Lattice<typename Action::FieldType, Action::dims>& field);

  public:
    MetropolisWorker() = default;

    void init();

    void run(uint steps);
};

}  // namespace reticolo::montecarlo
