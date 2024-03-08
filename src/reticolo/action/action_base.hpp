/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/action_base.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <cstddef>
#include <string>

#include "reticolo/lattice/lattice.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"

namespace reticolo::action {

template <typename FieldType, typename ActionType, size_t dim>
class ActionBase {
  public:
    /* Sync with lattice */
    virtual void lattice_sync(const Lattice<FieldType, dim>& field) = 0;

    /* Gloabal and local action computations */
    virtual auto compute_S(const Lattice<FieldType, dim>& field) -> ActionType = 0;
    virtual auto compute_S_loc(const Lattice<FieldType, dim>& field, int site) -> ActionType = 0;
    virtual auto compute_dS_loc(const Lattice<FieldType, dim>& field, const FieldType& dphi, int site)
        -> ActionType = 0;

    /* HMC methods */
    virtual void compute_Forces(const Lattice<FieldType, dim>& field, Lattice<FieldType, dim>& Forces) = 0;

    /* Log stuff */
    virtual auto action_name() -> std::string = 0;        // return the action name
    virtual auto action_parameters() -> std::string = 0;  // prints action parameters
};

}  // namespace reticolo::action
