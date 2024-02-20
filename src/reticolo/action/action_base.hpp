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
    virtual auto compute_S(const Lattice<FieldType, dim>& field) const -> ActionType = 0;
    virtual auto compute_S_loc(const Lattice<FieldType, dim>& field, const uintvect<dim>& coord) const
        -> ActionType = 0;
    virtual auto compute_dS_loc(const Lattice<FieldType, dim>& field, const FieldType& dphi,
                                const uintvect<dim>& coord) const -> ActionType = 0;

    // Log action information
    virtual auto action_name() -> std::string = 0;        // return the action name
    virtual auto action_parameters() -> std::string = 0;  // prints action parameters
};

}  // namespace reticolo::action
