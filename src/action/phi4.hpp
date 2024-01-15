#pragma once

#include "lattice/lattice.hpp"
#include "action/action_base.hpp"
#include "tools/types.hpp"

#include "H5Cpp.h"

#include <string> // std::string, std::stringstream
#include <format> // std::format()

namespace reticolo
{
    namespace action
    {
        template <ComplexValue FieldType, ComplexValue ActionType>
        class phi4 : public action_base<FieldType, ActionType, 4>
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

        private:
            params p;

        public:
            phi4(){};
            phi4(double l, double n, double u) : p(l, n, u){};
            phi4(params par) : p(par){};
            ~phi4(){};

            ActionType compute_S(const lattice<FieldType, 4> &field);
            ActionType compute_S_loc(const lattice<FieldType, 4> &field, const vect<4> &coord);
            ActionType compute_dS_loc(const lattice<FieldType, 4> &field, const FieldType &dphi, const vect<4> &coord);

            observables Measure(const lattice<FieldType, 4> &field);

            std::string action_name() { return "Complex phi^4"; };
            std::string action_parameters()
            {
                std::stringstream res;
                res << "lambda : " << std::format("{:4.1f}", p.lambda) << ", eta : " << std::format("{:4.1f}", p.eta) << ", mu : " << std::format("{:4.1f}", p.mu);
                return res.str();
            }
        };

        template <ComplexValue FieldType, ComplexValue ActionType>
        inline ActionType phi4<FieldType, ActionType>::compute_S(const lattice<FieldType, 4> &field)
        {
            double re = 0.0, im = 0.0;
            FieldType s;

            vect<4> sizes = field.getSizes();
            vect<4> coord;
            std::fill(coord.begin(), coord.end(), 0);

            for (uint i = 0; i < field.getNsites(); advance_vect(sizes, coord), i++)
            {
                s = field[coord];

                double phi2 = s.real() * s.real() + s.imag() * s.imag();

                re += 0.5 * p.eta * phi2                                                                                                  // (1/2) * (2d + m^2) phi_{a,x}^2
                      + 0.25 * p.lambda * phi2 * phi2                                                                                     // (lambda/4) * (phi_{a,x}^2)^2
                      - (s.real() * field[next(field, coord, _x)].real() + s.imag() * field[next(field, coord, _x)].imag())               // phi_{a,x} * phi_{a,x+1}
                      - (s.real() * field[next(field, coord, _y)].real() + s.imag() * field[next(field, coord, _y)].imag())               // phi_{a,x} * phi_{a,x+2}
                      - (s.real() * field[next(field, coord, _z)].real() + s.imag() * field[next(field, coord, _z)].imag())               // phi_{a,x} * phi_{a,x+3}
                      - cosh(p.mu) * (s.real() * field[next(field, coord, _t)].real() + s.imag() * field[next(field, coord, _t)].imag()); // cosh(mu) * phi_{a,x} * phi_{a,x+4}

                im += sinh(p.mu) * (s.real() * field[next(field, coord, _t)].imag() - s.imag() * field[next(field, coord, _t)].real()); // sinh(mu) * e_{a,b} * phi_{a,x} * phi_{b,x+4}
            }

            return ActionType(re, im);
        }

        template <ComplexValue FieldType, ComplexValue ActionType>
        inline ActionType phi4<FieldType, ActionType>::compute_S_loc(const lattice<FieldType, 4> &field, const vect<4> &coord)
        {
            FieldType
                s = field[coord],
                n_t = field[next(field, coord, _t)],
                p_t = field[prev(field, coord, _t)];

            double phi2 = s.real() * s.real() + s.imag() * s.imag();

            double re = 0.5 * p.eta * phi2              // (1/2) * (2d + m^2) phi_{a,x}^2
                        + 0.25 * p.lambda * phi2 * phi2 // (lambda/4) * (phi_{a,x}^2)^2
                        - s.real() * (field[next(field, coord, _x)].real() + field[prev(field, coord, _x)].real() +
                                      field[next(field, coord, _y)].real() + field[prev(field, coord, _y)].real() +
                                      field[next(field, coord, _z)].real() + field[prev(field, coord, _z)].real() +
                                      cosh(p.mu) * n_t.real() + cosh(p.mu) * p_t.real()) //
                        - s.imag() * (field[next(field, coord, _x)].imag() + field[prev(field, coord, _x)].imag() +
                                      field[next(field, coord, _y)].imag() + field[prev(field, coord, _y)].imag() +
                                      field[next(field, coord, _z)].imag() + field[prev(field, coord, _z)].imag() +
                                      cosh(p.mu) * n_t.imag() + cosh(p.mu) * p_t.imag());

            double im = sinh(p.mu) * (s.real() * n_t.imag() - s.imag() * n_t.real())    // sinh(mu) * e_{a,b} * phi_{a,x} * phi_{b,x+4}
                        + sinh(p.mu) * (p_t.real() * s.imag() - p_t.imag() * s.real()); // sinh(mu) * e_{a,b} * phi_{a,x-4} * phi_{b,x}

            return ActionType(re, im);
        }

        template <ComplexValue FieldType, ComplexValue ActionType>
        inline ActionType phi4<FieldType, ActionType>::compute_dS_loc(const lattice<FieldType, 4> &field, const FieldType &dphi, const vect<4> &coord)
        {
            FieldType
                s_old = field[coord],
                s_new = s_old + dphi,
                n_t = field[next(field, coord, _t)],
                p_t = field[prev(field, coord, _t)];

            double phi2_old = s_old.real() * s_old.real() + s_old.imag() * s_old.imag();
            double phi2_new = s_new.real() * s_new.real() + s_new.imag() * s_new.imag();

            double d_phi2 = phi2_new - phi2_old;
            double d_phi4 = phi2_new * phi2_new - phi2_old * phi2_old;

            ActionType neighbors_sum = static_cast<ActionType>(
                field[next(field, coord, _x)] + field[prev(field, coord, _x)]   //
                + field[next(field, coord, _y)] + field[prev(field, coord, _y)] //
                + field[next(field, coord, _z)] + field[prev(field, coord, _z)] //
                + FieldType(cosh(p.mu), 0.0) * (n_t + p_t));                    //

            double re = 0.5 * p.eta * d_phi2                  // delta( (1/2) * (2d + m^2) phi_{a,x}^2 )
                        + 0.25 * p.lambda * d_phi4            // delta( (lambda/4) * (phi_{a,x}^2)^2 )
                        - dphi.real() * neighbors_sum.real()  //
                        - dphi.imag() * neighbors_sum.imag(); //

            double im = sinh(p.mu) * (dphi.real() * n_t.imag() - dphi.imag() * n_t.real())    // sinh(mu) * e_{a,b} * phi_{a,x} * phi_{b,x+4}
                        + sinh(p.mu) * (p_t.real() * dphi.imag() - p_t.imag() * dphi.real()); // sinh(mu) * e_{a,b} * phi_{a,x-4} * phi_{b,x}

            return ActionType(re, im);
        }

        template <ComplexValue FieldType, ComplexValue ActionType>
        inline phi4<FieldType, ActionType>::observables phi4<FieldType, ActionType>::Measure(const lattice<FieldType, 4> &field)
        {
            FieldType phi;
            double phi2 = 0.0;

            vect<4> sizes = field.getSizes();
            vect<4> coord;
            std::fill(coord.begin(), coord.end(), 0);

            for (uint i = 0; i < field.getNsites(); advance_vect(sizes, coord), i++)
            {
                phi = field[coord];
                phi2 += phi.real() * phi.real() + phi.imag() * phi.imag();
            }

            return observables(0.5 * phi2 / (double)field.getNsites(), 0.0);
        }

    } // namespace action
} // namespace reticolo