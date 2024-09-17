/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/action_base.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <yaml-cpp/yaml.h>

#include <cstddef>
#include <string>

#include "reticolo/lattice/lattice.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"
#include "yaml-cpp/node/node.h"

namespace reticolo::action {

template <typename TField, typename TAction, RealValue TImpl>
class ActionBase {
  public:
    using field_type = TField;
    using action_type = TAction;

    /* Virtual destructor */
    virtual ~ActionBase() = default;

    virtual void setup(const YAML::Node&) = 0;

    /* Sync with lattice */
    virtual void lattice_sync(Lattice<field_type>&) = 0;

    /* Gloabal and local action computations */
    virtual auto compute_S(Lattice<field_type>&) -> action_type = 0;
    virtual auto compute_S_loc(Lattice<field_type>&, uint) -> action_type = 0;
    virtual auto compute_dS_loc(Lattice<field_type>&, const field_type&, uint) -> action_type = 0;

    /* HMC methods */
    virtual void compute_Forces(Lattice<field_type>&, Lattice<field_type>&) = 0;

    /* LLR methods */
    virtual void compute_LLRForces(Lattice<field_type>&, Lattice<field_type>&, TImpl, TImpl, TImpl) = 0;

    /* Log stuff */
    virtual auto GetName() -> std::string = 0;        // return the action name
    virtual auto GetParameters() -> std::string = 0;  // prints action parameters
};

}  // namespace reticolo::action
