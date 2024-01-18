/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/types/vect.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <iostream>
#include <array>

namespace reticolo
{
    // vector types
    template <size_t dim>
    using uintvect = std::array<uint, dim>;

    template <size_t dim>
    uint operator*(const uintvect<dim> &lhs, const uintvect<dim> &rhs)
    {
        uint res = 0;
        for (int i = 0; i < dim; i++)
            res += lhs[i] * rhs[i];
        return res;
    }

    template <size_t dim>
    inline void advance_coord(const uintvect<dim> &sizes, uintvect<dim> &coord)
    {
        coord.back()++;
        for (int i = dim - 1; i >= 0; i--)
            if (coord[i] == sizes[i])
            {
                coord[i] = 0;
                coord[i - 1]++;
            }
            else
                return;
    }

    template <size_t dim>
    inline void print(const uintvect<dim> &v)
    {
        for (int i = 0; i < dim; i++)
            std::cout << v[i];
        std::cout << std::endl;
    }
} // namespace reticolo