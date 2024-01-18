/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: montecarlo/impl/metropolis.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#include "montecarlo/metropolis.hpp"

#include "lattice/lattice.hpp"
#include "tools/types.hpp"

#include "H5Cpp.h"

namespace reticolo
{
    namespace montecarlo
    {
        // Real actions
        template <template <typename, typename> class Action, typename FieldType, typename ActionType, size_t dim>
        inline H5::CompType Markov_worker<Action, FieldType, ActionType, dim>::MC_data::get_hdf5_CompType()
            requires RealValue<FieldType>
        {
            H5::CompType type;
            type.insertMember("acceptance", HOFFSET(MC_data, acceptance), H5::PredType::NATIVE_DOUBLE);
            type.insertMember("S", HOFFSET(MC_data, S), H5::PredType::NATIVE_DOUBLE);
            type.insertMember("dS", HOFFSET(MC_data, dS), H5::PredType::NATIVE_DOUBLE);
            return type;
        }

        // Complex actions
        template <template <typename, typename> class Action, typename FieldType, typename ActionType, size_t dim>
        inline H5::CompType Markov_worker<Action, FieldType, ActionType, dim>::MC_data::get_hdf5_CompType()
            requires ComplexValue<FieldType>
        {
            H5::CompType type;
            type.insertMember("acceptance", HOFFSET(MC_data, acceptance), H5::PredType::NATIVE_DOUBLE);
            type.insertMember("S_re", HOFFSET(MC_data, S), H5::PredType::NATIVE_DOUBLE);
            type.insertMember("S_im", HOFFSET(MC_data, S + sizeof<FieldType> / 2), H5::PredType::NATIVE_DOUBLE);
            type.insertMember("dS_re", HOFFSET(MC_data, dS), H5::PredType::NATIVE_DOUBLE);
            type.insertMember("dS_im", HOFFSET(MC_data, dS.+ sizeof<FieldType> / 2), H5::PredType::NATIVE_DOUBLE);
            return type;
        }

        template <template <typename, typename> class Action, typename FieldType, typename ActionType, size_t dim>
        inline void Markov_worker<Action, FieldType, ActionType, dim>::sweep(lattice<FieldType, dim> &field)
        {
            uint acc = 0;

            ActionType
                dS,               // local action variation
                dS_tot(0.0, 0.0); // cumulative action variation

            FieldType dfield; // local field variation

            uintvect<dim> sizes = field.getSizes();
            uintvect<dim> coord;
            std::fill(coord.begin(), coord.end(), 0);

            for (uint i = 0; i < field.getNsites(); advance_coord(sizes, coord), i++)
            {
                // dfield = FieldType(0.2 * norm(rng), 0.2 * norm(rng));
                // dfield = FieldType(0.2 * (unif(rng) - 0.5), 0.2 * (unif(rng) - 0.5));
                dfiel = 0.2 * (random(unif, rng));

                dS = action.compute_dS_loc(field, dfield, coord);

                if (exp(-dS.real()) > unif(rng))
                {
                    acc++;
                    field[coord] += dfield;
                    dS_tot += dS;
                }
            }

            MC_stats.acceptance = static_cast<double>(acc) / field.getNsites();
            MC_stats.S_re += dS_tot.real();
            MC_stats.S_im += dS_tot.imag();
            MC_stats.dS_re = dS_tot.real();
            MC_stats.dS_im = dS_tot.imag();
        }

    } // namespace montecarlo

} // namespace reticolo
