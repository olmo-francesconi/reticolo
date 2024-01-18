/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/types/basic.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <complex>

namespace reticolo
{
    // enumerator for lattice dimensions
    enum
    {
        _t = 0,
        _x = 1,
        _y = 2,
        _z = 3
    };

    // enumerator for lattice dimensions
    enum
    {
        _x0 = 0,
        _x1 = 1,
        _x2 = 2,
        _x3 = 3,
        _x4 = 4,
        _x5 = 5,
        _x6 = 6,
        _x7 = 7,
        _x8 = 8,
        _x9 = 9,
    };

    // basic types
    typedef unsigned int uint;
    typedef float RealF;
    typedef double RealD;
    typedef std::complex<float> ComplexF;
    typedef std::complex<double> ComplexD;
}