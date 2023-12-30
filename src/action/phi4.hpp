#pragma once

#include "lattice/lattice.hpp"
#include "action/action_base.hpp"
#include "tools/types.hpp"

#include <string> // std::string, std::stringstream
#include <format> // std::format()

namespace LLR
{
    namespace action
    {
        class phi4 : public action_base<ComplexD, ComplexD>
        {
        private:
            double lambda;
            double eta;
            double mu;

        public:
            phi4() : lambda(1.0), eta(9.0), mu(0.0){};
            phi4(double l, double n, double u) : lambda(l), eta(n), mu(u){};

            ComplexD compute_S(const lattice<ComplexD> &field);
            ComplexD compute_S_loc(const lattice<ComplexD> &field, const vect4 &coord);
            ComplexD compute_dS_loc(const lattice<ComplexD> &field, const ComplexD &dphi, const vect4 &coord);

            std::string action_name() { return "Complex phi^4"; };
            std::string action_parameters()
            {
                std::stringstream res;
                res << "lambda : " << std::format("{:4.1f}", lambda)
                    << ", eta : " << std::format("{:4.1f}", eta)
                    << ", mu : " << std::format("{:4.1f}", mu);
                return res.str();
            }
            ~phi4(){};
        };

        inline ComplexD phi4::compute_S(const lattice<ComplexD> &field)
        {
            double re = 0.0, im = 0.0;
            ComplexD s, n_t, n_x, n_y, n_z;

            vect4 sizes = field.getSizes();
            vect4 coord;

            for (coord[_t] = 0; coord[_t] < sizes[_t]; coord[_t]++)
                for (coord[_x] = 0; coord[_x] < sizes[_x]; coord[_x]++)
                    for (coord[_y] = 0; coord[_y] < sizes[_y]; coord[_y]++)
                        for (coord[_z] = 0; coord[_z] < sizes[_z]; coord[_z]++)
                        {
                            s = field[coord];
                            n_t = field[nt(field, coord)];
                            n_x = field[nx(field, coord)];
                            n_y = field[ny(field, coord)];
                            n_z = field[nz(field, coord)];

                            double phi2 = s.real() * s.real() + s.imag() * s.imag();

                            re += 0.5 * eta * phi2                                              // (1/2) * (2d + m^2) phi_{a,x}^2
                                  + 0.25 * lambda * phi2 * phi2                                 // (lambda/4) * (phi_{a,x}^2)^2
                                  - (s.real() * n_x.real() + s.imag() * n_x.imag())             // phi_{a,x} * phi_{a,x+1}
                                  - (s.real() * n_y.real() + s.imag() * n_y.imag())             // phi_{a,x} * phi_{a,x+2}
                                  - (s.real() * n_z.real() + s.imag() * n_z.imag())             // phi_{a,x} * phi_{a,x+3}
                                  - cosh(mu) * (s.real() * n_t.real() + s.imag() * n_t.imag()); // cosh(mu) * phi_{a,x} * phi_{a,x+4}

                            im += sinh(mu) * (s.real() * n_t.imag() - s.imag() * n_t.real()); // sinh(mu) * e_{a,b} * phi_{a,x} * phi_{b,x+4}
                        }

            return ComplexD(re, im);
        }

        inline ComplexD phi4::compute_S_loc(const lattice<ComplexD> &field, const vect4 &coord)
        {
            ComplexD
                s = field[coord],
                n_t = field[nt(field, coord)],
                n_x = field[nx(field, coord)],
                n_y = field[ny(field, coord)],
                n_z = field[nz(field, coord)],
                p_t = field[pt(field, coord)],
                p_x = field[px(field, coord)],
                p_y = field[py(field, coord)],
                p_z = field[pz(field, coord)];

            double phi2 = s.real() * s.real() + s.imag() * s.imag();

            double re = 0.5 * eta * phi2                                              // (1/2) * (2d + m^2) phi_{a,x}^2
                        + 0.25 * lambda * phi2 * phi2                                 // (lambda/4) * (phi_{a,x}^2)^2
                        - (s.real() * n_x.real() + s.imag() * n_x.imag())             // phi_{a,x} * phi_{a,x+1}
                        - (s.real() * p_x.real() + s.imag() * p_x.imag())             // phi_{a,x} * phi_{a,x-1}
                        - (s.real() * n_y.real() + s.imag() * n_y.imag())             // phi_{a,x} * phi_{a,x+2}
                        - (s.real() * p_y.real() + s.imag() * p_y.imag())             // phi_{a,x} * phi_{a,x-2}
                        - (s.real() * n_z.real() + s.imag() * n_z.imag())             // phi_{a,x} * phi_{a,x+3}
                        - (s.real() * p_z.real() + s.imag() * p_z.imag())             // phi_{a,x} * phi_{a,x-3}
                        - cosh(mu) * (s.real() * n_t.real() + s.imag() * n_t.imag())  // cosh(mu) * phi_{a,x} * phi_{a,x+4}
                        - cosh(mu) * (s.real() * p_t.real() + s.imag() * p_t.imag()); // cosh(mu) * phi_{a,x} * phi_{a,x-4}

            double im = sinh(mu) * (s.real() * n_t.imag() - s.imag() * n_t.real())    // sinh(mu) * e_{a,b} * phi_{a,x} * phi_{b,x+4}
                        + sinh(mu) * (p_t.real() * s.imag() - p_t.imag() * s.real()); // sinh(mu) * e_{a,b} * phi_{a,x-4} * phi_{b,x}

            return ComplexD(re, im);
        }

        inline ComplexD phi4::compute_dS_loc(const lattice<ComplexD> &field, const ComplexD &dphi, const vect4 &coord)
        {

            ComplexD
                s_old = field[coord],
                s_new = s_old + dphi,
                n_t = field[nt(field, coord)],
                p_t = field[pt(field, coord)];

            double phi2_old = s_old.real() * s_old.real() + s_old.imag() * s_old.imag();
            double phi2_new = s_new.real() * s_new.real() + s_new.imag() * s_new.imag();

            double d_phi2 = phi2_new - phi2_old;
            double d_phi4 = phi2_new * phi2_new - phi2_old * phi2_old;

            ComplexD neighbors_sum = field[nx(field, coord)] + field[px(field, coord)]   //
                                     + field[ny(field, coord)] + field[py(field, coord)] //
                                     + field[nz(field, coord)] + field[pz(field, coord)] //
                                     + cosh(mu) * (n_t + p_t);                           //

            double re = 0.5 * eta * d_phi2                    // delta( (1/2) * (2d + m^2) phi_{a,x}^2 )
                        + 0.25 * lambda * d_phi4              // delta( (lambda/4) * (phi_{a,x}^2)^2 )
                        - dphi.real() * neighbors_sum.real()  //
                        - dphi.imag() * neighbors_sum.imag(); //

            double im = sinh(mu) * (dphi.real() * n_t.imag() - dphi.imag() * n_t.real())    // sinh(mu) * e_{a,b} * phi_{a,x} * phi_{b,x+4}
                        + sinh(mu) * (p_t.real() * dphi.imag() - p_t.imag() * dphi.real()); // sinh(mu) * e_{a,b} * phi_{a,x-4} * phi_{b,x}

            return ComplexD(re, im);
        }

    } // namespace action
} // namespace LLR