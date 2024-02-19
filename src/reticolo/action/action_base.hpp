/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/action_base.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <string>

#include "reticolo/lattice/lattice.hpp"
#include "reticolo/tools/types/concepts.hpp"

namespace reticolo::action {

template <typename FieldType, typename ActionType, size_t dim>
class action_base {
 public:
  struct observables;

  virtual auto compute_S(const lattice<FieldType, dim>& field) const -> ActionType = 0;
  virtual auto compute_S_loc(const lattice<FieldType, dim>& field, const uintvect<dim>& coord) const -> ActionType = 0;
  virtual auto compute_dS_loc(const lattice<FieldType, dim>& field, const FieldType& dphi,
                              const uintvect<dim>& coord) const -> ActionType = 0;

  // virtual observables Measure(const lattice<FieldType, dim> &field) = 0;

  // Log action information
  virtual auto action_name() -> std::string = 0;        // return the action name
  virtual auto action_parameters() -> std::string = 0;  // prints action parameters
};

}  // namespace reticolo::action
