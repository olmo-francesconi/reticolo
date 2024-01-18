/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/types/hfield.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <iostream>
#include <array>

// look-up table for the symmetric h_\mu\nu components
// These speed-up data access by referencing directly the data in the linearized
// 10 component array insted of compiting the index from the mu nu components.
#define MUNU_00 0
#define MUNU_01 1
#define MUNU_02 2
#define MUNU_03 3
#define MUNU_10 1
#define MUNU_11 4
#define MUNU_12 5
#define MUNU_13 6
#define MUNU_20 2
#define MUNU_21 5
#define MUNU_22 7
#define MUNU_23 8
#define MUNU_30 3
#define MUNU_31 6
#define MUNU_32 8
#define MUNU_33 9

namespace reticolo
{
    // Basic matrix class to store h fields
    struct HField
    {
        std::array<double, 10> mat;

        HField(){};  // std::cout << "HField::HField() constructor called" << std::endl; };
        ~HField(){}; // std::cout << "HField::~HField() destructor called" << std::endl; };

        // Array-like access operators
        double &operator[](size_t index) { return mat[index]; };
        const double &operator[](size_t index) const { return mat[index]; };
        double &operator()(size_t mu, size_t nu)
        {
            if (nu < mu)
                std::swap(mu, nu);
            return mat[4 * mu + nu - (mu * (mu + 1)) / 2];
        }
        const double &operator()(size_t mu, size_t nu) const
        {
            if (nu < mu)
                std::swap(mu, nu);
            return mat[4 * mu + nu - (mu * (mu + 1)) / 2];
        }

        HField &operator=(const HField &other)
        {
            std::copy(other.mat.begin(), other.mat.end(), mat.begin());
            return *this;
        }

        double trace() const;

        void print();
    }; // struct HField

    inline double HField::trace() const
    {
        return mat[0] + mat[4] + mat[7] + mat[9];
    }

    inline void HField::print()
    {
        std::cout << std::scientific;
        for (int mu = 0; mu < 4; mu++)
        {
            for (int nu = 0; nu < 4; nu++)
                std::cout << (*this)(mu, nu) << "\t";
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }

    // Math methods for Hfield objects
    namespace HField_math
    {
        // compute res = c * (A - B) element wise
        inline void diff(HField &res, const HField &A, const HField &B, const double c = 1.0)
        {
            for (size_t i = 0; i < 10; i++)
                res[i] = c * (A[i] - B[i]);
        }

        // compute res = c * (A + B) element wise
        inline void sum(HField &res, const HField &A, const HField &B, const double c = 1.0)
        {
            for (size_t i = 0; i < 10; i++)
                res[i] = c * (A[i] + B[i]);
        }

        inline void prod(HField &res, const double lhs, const HField &rhs)
        {
            for (size_t i = 0; i < 10; i++)
                res[i] = lhs * rhs[i];
        }

    } // namespace HField_math

} // namespace reticolo