#pragma once

#include <array>

namespace reticolo
{
    // vector types
    template <size_t dim>
    using vect = std::array<uint, dim>;

    template <size_t dim>
    uint operator*(const vect<dim> &lhs, const vect<dim> &rhs)
    {
        uint res = 0;
        for (int i = 0; i < dim; i++)
            res += lhs[i] * rhs[i];
        return res;
    }

    template <size_t dim>
    inline void advance_vect(const vect<dim> &sizes, vect<dim> &coord)
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
} // namespace reticolo