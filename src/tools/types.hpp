#pragma once

#include <array>

namespace LLR
{
    typedef unsigned int uint;

    typedef std::array<uint, 4> vect4;

    enum
    {
        _t = 0,
        _x = 1,
        _y = 2,
        _z = 3
    };
}