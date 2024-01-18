/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: action/impl/phi4.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#include "action/phi4.hpp"

#include "lattice/lattice.hpp"
#include "tools/types.hpp"

#include "H5Cpp.h"

#include <string>
#include <format>

namespace reticolo
{
    namespace action
    {
        ComplexD phi4::compute_S(const lattice<ComplexD, 4> &field)
        {
            double re = 0.0, im = 0.0;
            ComplexD s;

            uintvect<4> sizes = field.getSizes();
            uintvect<4> coord;
            std::fill(coord.begin(), coord.end(), 0);

            for (uint i = 0; i < field.getNsites(); advance_coord(sizes, coord), i++)
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

            return ComplexD(re, im);
        };

        ComplexD phi4::compute_S_loc(const lattice<ComplexD, 4> &field, const uintvect<4> &coord)
        {
            ComplexD
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

            return ComplexD(re, im);
        };

        ComplexD phi4::compute_dS_loc(const lattice<ComplexD, 4> &field, const ComplexD &dphi, const uintvect<4> &coord)
        {
            ComplexD
                s_old = field[coord],
                s_new = s_old + dphi,
                n_t = field[next(field, coord, _t)],
                p_t = field[prev(field, coord, _t)];

            double phi2_old = s_old.real() * s_old.real() + s_old.imag() * s_old.imag();
            double phi2_new = s_new.real() * s_new.real() + s_new.imag() * s_new.imag();

            double d_phi2 = phi2_new - phi2_old;
            double d_phi4 = phi2_new * phi2_new - phi2_old * phi2_old;

            ComplexD neighbors_sum =
                field[next(field, coord, _x)] + field[prev(field, coord, _x)]   //
                + field[next(field, coord, _y)] + field[prev(field, coord, _y)] //
                + field[next(field, coord, _z)] + field[prev(field, coord, _z)] //
                + cosh(p.mu) * (n_t + p_t);                                     //

            double re = 0.5 * p.eta * d_phi2                  // delta( (1/2) * (2d + m^2) phi_{a,x}^2 )
                        + 0.25 * p.lambda * d_phi4            // delta( (lambda/4) * (phi_{a,x}^2)^2 )
                        - dphi.real() * neighbors_sum.real()  //
                        - dphi.imag() * neighbors_sum.imag(); //

            double im = (dphi.real() * n_t.imag() - dphi.imag() * n_t.real())    // e_{a,b} * phi_{a,x} * phi_{b,x+4}
                        + (p_t.real() * dphi.imag() - p_t.imag() * dphi.real()); // e_{a,b} * phi_{a,x-4} * phi_{b,x}

            return ComplexD(re, im);
        };

        phi4::observables phi4::Measure(const lattice<ComplexD, 4> &field)
        {
            ComplexD phi;
            double phi2 = 0.0;

            uintvect<4> sizes = field.getSizes();
            uintvect<4> coord;
            std::fill(coord.begin(), coord.end(), 0);

            for (uint i = 0; i < field.getNsites(); advance_coord(sizes, coord), i++)
            {
                phi = field[coord];
                phi2 += phi.real() * phi.real() + phi.imag() * phi.imag();
            }

            return observables(0.5 * phi2 / (double)field.getNsites(), 0.0);
        };

    } // namespace action

} // namespace reticolo