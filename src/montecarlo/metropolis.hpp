/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: montecarlo/metropolis.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include "lattice/lattice.hpp"
#include "tools/types.hpp"

#include "H5Cpp.h"

namespace reticolo
{
    namespace montecarlo
    {
        template <template <typename, typename> class Action, typename FieldType, typename ActionType, size_t dim>
        class Markov_worker
        {
        public:
            struct MC_data
            {
                double
                    acceptance;
                ActionType S, dS;

                std::string dump_str() { return std::format("{:+8e},", acceptance) + print(S) + "," + print(dS); }
                H5::CompType get_hdf5_CompType()
                    requires RealValue<FieldType>;
                H5::CompType get_hdf5_CompType()
                    requires ComplexValue<FieldType>;
            };
            void make_hdf5_CompType(H5::CompType &type)
            {
                type.insertMember("acceptance", HOFFSET(MC_data, acceptance), H5::PredType::NATIVE_DOUBLE);
                type.insertMember("S_re", HOFFSET(MC_data, S_re), H5::PredType::NATIVE_DOUBLE);
                type.insertMember("S_im", HOFFSET(MC_data, S_im), H5::PredType::NATIVE_DOUBLE);
                type.insertMember("dS_re", HOFFSET(MC_data, dS_re), H5::PredType::NATIVE_DOUBLE);
                type.insertMember("dS_im", HOFFSET(MC_data, dS_im), H5::PredType::NATIVE_DOUBLE);
            }

        private:
            void sweep(lattice<FieldType, dim> &field);

        public:
            Markov_worker(){};

            void init();

            void run(uint steps);
        };

    } // namespace montecarlo

} // namespace reticolo