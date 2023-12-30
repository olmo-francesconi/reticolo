#pragma once

#include <complex>
#include <array>

namespace LLR
{
    // basic types
    typedef unsigned int uint;
    typedef float RealF;
    typedef double RealD;
    typedef std::complex<float> ComplexF;
    typedef std::complex<double> ComplexD;

    // vector types
    typedef std::array<uint, 4> vect4;

    // enumerator for lattice dimensions
    enum
    {
        _t = 0,
        _x = 1,
        _y = 2,
        _z = 3
    };
}