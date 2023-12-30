#pragma once

#include "lattice/lattice.hpp"
#include "tools/types.hpp"

#include <string>

namespace LLR
{
    namespace action
    {
        template <typename T_field, typename T_action>
        class action_base
        {
        public:
            virtual T_action compute_S(const lattice<T_field> &field) = 0;
            virtual T_action compute_S_loc(const lattice<T_field> &field, const vect4 &coord) = 0;
            virtual T_action compute_dS_loc(const lattice<T_field> &field, const T_field &dphi, const vect4 &coord) = 0;

            // Log action information
            virtual std::string action_name() = 0;       // return the action name
            virtual std::string action_parameters() = 0; // prints action parameters
        };
    } // namespace action
} // namespace LLR