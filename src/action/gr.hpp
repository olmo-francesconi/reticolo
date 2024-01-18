/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/gr.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include "action/action_base.hpp"
#include "tools/types.hpp"

#include "H5Cpp.h"

#include <string> // std::string, std::stringstream
#include <format> // std::format()

namespace reticolo
{
    namespace action
    {

        class WeakFieldEuclideanGR : action_base<HField, RealD, 4>
        {
        public:
            struct params
            {
                double lambda;
                double eta;
                double mu;

                params() : lambda(1.0), eta(9.0), mu(0){};
                params(double l, double e, double u) : lambda(l), eta(e), mu(u){};
            };

            struct observables
            {
                double R;

                std::string dump_str() { return std::format("{:+8e}", R); }
                std::vector<double> dump_data() const { return std::vector<double>({R}); }
            };
            void make_hdf5_CompType(H5::CompType &type)
            {
                type.insertMember("R", HOFFSET(observables, R), H5::PredType::NATIVE_DOUBLE);
            }

        private:
        public:
            virtual RealD compute_S(const lattice<HField, 4> &field);
            virtual RealD compute_S_loc(const lattice<HField, 4> &field, const uintvect<4> &coord);
            virtual RealD compute_dS_loc(const lattice<HField, 4> &field, const HField &dphi, const uintvect<4> &coord);

            virtual void Measure(const lattice<HField, 4> &field);

            // Log action information
            virtual std::string action_name() = 0;       // return the action name
            virtual std::string action_parameters() = 0; // prints action parameters
        };
    } // namespace action
} // namespace reticolo