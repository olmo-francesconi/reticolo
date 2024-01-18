/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/action_base.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include "lattice/lattice.hpp"
#include "tools/types.hpp"

#include <string>

namespace reticolo
{
    namespace action
    {
        template <typename FieldType, typename ActionType, size_t dim>
        class action_base
        {
        public:
            struct observables;

            virtual ActionType compute_S(const lattice<FieldType, dim> &field) = 0;
            virtual ActionType compute_S_loc(const lattice<FieldType, dim> &field, const uintvect<dim> &coord) = 0;
            virtual ActionType compute_dS_loc(const lattice<FieldType, dim> &field, const FieldType &dphi, const uintvect<dim> &coord) = 0;

            // virtual observables Measure(const lattice<FieldType, dim> &field) = 0;

            // Log action information
            virtual std::string action_name() = 0;       // return the action name
            virtual std::string action_parameters() = 0; // prints action parameters
        };
    } // namespace action

} // namespace reticolo