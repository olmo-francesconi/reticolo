/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/action_base.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <yaml-cpp/yaml.h>

#include <cstddef>
#include <string>

#include "reticolo/lattice/Lattice.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "yaml-cpp/node/node.h"

namespace reticolo::action {

template <typename FieldT, typename ActionT>
class ActionBase {
  public:
    using FieldType = FieldT;
    using ActionType = ActionT;

    /* Virtual destructor */
    virtual ~ActionBase() = default;

    virtual void setup(const YAML::Node& ActionParamsDict) = 0;

    /* Sync with lattice */
    virtual void lattice_sync(Lattice<FieldType>& field) = 0;

    /* Gloabal and local action computations */
    virtual auto compute_S(Lattice<FieldType>& field) -> ActionType = 0;
    virtual auto compute_S_loc(Lattice<FieldType>& field, int site) -> ActionType = 0;
    virtual auto compute_dS_loc(Lattice<FieldType>& field, const FieldType& dphi, int site) -> ActionType = 0;

    /* HMC methods */
    virtual void compute_Forces(Lattice<FieldType>& field, Lattice<FieldType>& forces) = 0;

    /* LLR methods */
    virtual void compute_LLRForces(Lattice<FieldType>& field, Lattice<FieldType>& forces, double Sk, double width,
                                   double ak) = 0;

    /* Log stuff */
    virtual auto GetName() -> std::string = 0;        // return the action name
    virtual auto GetParameters() -> std::string = 0;  // prints action parameters
};

}  // namespace reticolo::action
