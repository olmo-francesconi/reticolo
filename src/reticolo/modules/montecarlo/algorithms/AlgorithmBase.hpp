/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: factory/AlgorithmBase.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include "reticolo/lattice/Lattice.hpp"
#include "reticolo/modules/montecarlo/MonteCarloData.hpp"
#include "yaml-cpp/node/node.h"

namespace reticolo {

template <class Action>
class AlgorithmBase {
    using action_type = Action::ActionType;
    using field_type = typename Action::FieldType;
    using observables_type = typename Action::Observables;
    using monte_carlo_data_type = montecarlo::data<typename Action::ActionType>;

  public:
    /* virtual destructor */
    virtual ~AlgorithmBase() = default;

    /* setup */
    virtual void setup(const YAML::Node&, const Lattice<field_type>&) = 0;

    /* execution */
    virtual void updateField(Lattice<field_type>&, Action&, monte_carlo_data_type&) = 0;
};

}  // namespace reticolo
