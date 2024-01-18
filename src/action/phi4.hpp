/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/phi4.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include "lattice/lattice.hpp"
// #include "action/action_base.hpp"
#include "tools/types.hpp"

#include "H5Cpp.h"

#include <string>
#include <format>

namespace reticolo
{
    namespace action
    {
        class phi4
        {
        public:
            using FieldType = ComplexD;
            using ActionType = ComplexD;
            const static int dims = 4;

            struct params
            {
                double lambda;
                double eta;
                double mu;

                params() : lambda(1.0), eta(9.0), mu(0){};
                params(double l, double e, double u) : lambda(l), eta(e), mu(u){};
            } p;

            struct observables
            {
                double phi2;
                double density;

                std::string dump_str() { return std::format("{:+8e},{:+8e}", phi2, density); }
                std::vector<double> dump_data() const { return std::vector<double>({phi2, density}); }
            };
            void make_hdf5_CompType(H5::CompType &type)
            {
                type.insertMember("phi2", HOFFSET(observables, phi2), H5::PredType::NATIVE_DOUBLE);
                type.insertMember("density", HOFFSET(observables, density), H5::PredType::NATIVE_DOUBLE);
            }

            phi4(){};
            phi4(double l, double n, double u) : p(l, n, u){};
            phi4(params par) : p(par){};
            ~phi4(){};

            ComplexD compute_S(const lattice<ComplexD, 4> &field);
            ComplexD compute_S_loc(const lattice<ComplexD, 4> &field, const uintvect<4> &coord);
            ComplexD compute_dS_loc(const lattice<ComplexD, 4> &field, const ComplexD &dphi, const uintvect<4> &coord);

            observables Measure(const lattice<ComplexD, 4> &field);

            std::string action_name() { return "Complex phi^4"; };
            std::string action_parameters()
            {
                std::stringstream res;
                res << "lambda : " << std::format("{:4.1f}", p.lambda) << ", eta : " << std::format("{:4.1f}", p.eta) << ", mu : " << std::format("{:4.1f}", p.mu);
                return res.str();
            }
        };

    } // namespace action

} // namespace reticolo