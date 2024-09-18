/*******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: factory/AlgorithmBase.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

*******************************************************************************/

#pragma once

#include <random>

#include "reticolo/lattice/lattice.hpp"
#include "reticolo/modules/montecarlo/MonteCarloData.hpp"
#include "yaml-cpp/node/node.h"

namespace reticolo::MMonteCarlo {

template <class Action, class TGen = std::mt19937_64>
class MCAlgorithmBase {
  public:
    using field_type = Action::field_type;
    using monte_carlo_data_type = MMonteCarlo::data<typename Action::action_type>;

    /* virtual destructor */
    virtual ~MCAlgorithmBase() = default;

    /* setup */
    virtual void setup(const YAML::Node&, const Lattice<field_type>&) = 0;

    /* execution */
    virtual void updateField(Lattice<field_type>&, Action&, monte_carlo_data_type&, TGen&) = 0;
};

}  // namespace reticolo::MMonteCarlo
