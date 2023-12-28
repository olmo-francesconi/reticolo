#pragma once

#include "lattice/lattice.hpp"
#include "tools/types.hpp"

#include <numeric>
#include <iostream>

namespace LLR
{
    namespace action
    {
        struct phi4_params
        {
            double
                lambda, // coupling
                eta,    // mass parameter
                mu;     // chemical potential
        };

        class phi4
        {
        private:
            phi4_params p;
            lattice<std::complex<double>> *field;

        public:
            phi4(){};
            phi4(lattice<std::complex<double>> *field, phi4_params params) { init(field, params); };

            void init(lattice<std::complex<double>> *latt, phi4_params params) { field = latt, p = params; };

            std::complex<double> compute_S() const;
            std::complex<double> compute_S_loc(vect4 &coord) const;
        };

        inline std::complex<double> phi4::compute_S() const
        {
            double re = 0.0, im = 0.0;
            std::complex<double> s, n_t, n_x, n_y, n_z;

            vect4 sizes = (*field).getSizes();
            vect4 coord;

            for (coord[_t] = 0; coord[_t] < sizes[_t]; coord[_t]++)
                for (coord[_x] = 0; coord[_x] < sizes[_x]; coord[_x]++)
                    for (coord[_y] = 0; coord[_y] < sizes[_y]; coord[_y]++)
                        for (coord[_z] = 0; coord[_z] < sizes[_z]; coord[_z]++)
                        {
                            s = (*field)[coord];
                            n_t = (*field)[nt((*field), coord)];
                            n_x = (*field)[nx((*field), coord)];
                            n_y = (*field)[ny((*field), coord)];
                            n_z = (*field)[nz((*field), coord)];

                            double phi2 = s.real() * s.real() + s.imag() * s.imag();

                            re += 0.5 * p.eta * phi2                                              // (1/2) * (2d + m^2) phi_{a,x}^2
                                  + 0.25 * p.lambda * phi2 * phi2                                 // (lambda/4) * (phi_{a,x}^2)^2
                                  - (s.real() * n_x.real() + s.imag() * n_x.imag())               // phi_{a,x} * phi_{a,x+1}
                                  - (s.real() * n_y.real() + s.imag() * n_y.imag())               // phi_{a,x} * phi_{a,x+2}
                                  - (s.real() * n_z.real() + s.imag() * n_z.imag())               // phi_{a,x} * phi_{a,x+3}
                                  - cosh(p.mu) * (s.real() * n_t.real() + s.imag() * n_t.imag()); // cosh(mu) * phi_{a,x} * phi_{a,x+4}

                            im += sinh(p.mu) * (s.real() * n_t.imag() - s.imag() * n_t.real()); // sinh(mu) * e_{a,b} * phi_{a,x} * phi_{b,x+4}
                        }

            return std::complex<double>(re, im);
        }

        inline std::complex<double> phi4::compute_S_loc(vect4 &coord) const
        {
            std::complex<double>
                s = (*field)[coord],
                n_t = (*field)[nt((*field), coord)],
                n_x = (*field)[nx((*field), coord)],
                n_y = (*field)[ny((*field), coord)],
                n_z = (*field)[nz((*field), coord)],
                p_t = (*field)[pt((*field), coord)],
                p_x = (*field)[px((*field), coord)],
                p_y = (*field)[py((*field), coord)],
                p_z = (*field)[pz((*field), coord)];

            double phi2 = s.real() * s.real() + s.imag() * s.imag();

            double re = 0.5 * p.eta * phi2                                              // (1/2) * (2d + m^2) phi_{a,x}^2
                        + 0.25 * p.lambda * phi2 * phi2                                 // (lambda/4) * (phi_{a,x}^2)^2
                        - (s.real() * n_x.real() + s.imag() * n_x.imag())               // phi_{a,x} * phi_{a,x+1}
                        - (s.real() * p_x.real() + s.imag() * p_x.imag())               // phi_{a,x} * phi_{a,x-1}
                        - (s.real() * n_y.real() + s.imag() * n_y.imag())               // phi_{a,x} * phi_{a,x+2}
                        - (s.real() * p_y.real() + s.imag() * p_y.imag())               // phi_{a,x} * phi_{a,x-2}
                        - (s.real() * n_z.real() + s.imag() * n_z.imag())               // phi_{a,x} * phi_{a,x+3}
                        - (s.real() * p_z.real() + s.imag() * p_z.imag())               // phi_{a,x} * phi_{a,x-3}
                        - cosh(p.mu) * (s.real() * n_t.real() + s.imag() * n_t.imag())  // cosh(mu) * phi_{a,x} * phi_{a,x+4}
                        - cosh(p.mu) * (s.real() * p_t.real() + s.imag() * p_t.imag()); // cosh(mu) * phi_{a,x} * phi_{a,x-4}

            double im = sinh(p.mu) * (s.real() * n_t.imag() - s.imag() * n_t.real())    // sinh(mu) * e_{a,b} * phi_{a,x} * phi_{b,x+4}
                        + sinh(p.mu) * (p_t.real() * s.imag() - p_t.imag() * s.real()); // sinh(mu) * e_{a,b} * phi_{a,x-4} * phi_{b,x}

            return std::complex<double>(re, im);
        }

    } // namespace action
} // namespace LLR