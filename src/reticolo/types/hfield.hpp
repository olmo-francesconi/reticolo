/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/types/hfield.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <H5Ipublic.h>
#include <H5Tpublic.h>
#include <H5public.h>
#include <H5version.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <iostream>
#include <utility>

#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"

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

namespace reticolo {
// Basic matrix class to store h fields
template <RealValue T>
struct HField {
    static const size_t NumComp = 10;

    std::array<T, NumComp> _Mat;

    HField() = default;
    HField(T val) { std::fill(_Mat.begin(), _Mat.end(), val); }

    ~HField() = default;

    // Array-like access operators
    auto operator[](size_t index) -> T& { return _Mat[index]; };
    auto operator[](size_t index) const -> const T& { return _Mat[index]; };
    auto operator()(size_t mu, size_t nu) -> T& {
        if (nu < mu) {
            std::swap(mu, nu);
        }
        return _Mat[4 * mu + nu - (mu * (mu + 1)) / 2];
    }

    auto operator()(size_t mu, size_t nu) const -> const T& {
        if (nu < mu) {
            std::swap(mu, nu);
        }
        return _Mat[4 * mu + nu - (mu * (mu + 1)) / 2];
    }

    auto operator=(const HField& other) -> HField& {
        std::copy(other._Mat.begin(), other._Mat.end(), _Mat.begin());
        return *this;
    }

    auto operator+=(const HField& other) -> HField& {
        for (size_t Comp = 0; Comp < NumComp; Comp++) {
            this->_Mat[Comp] += other._Mat[Comp];
        }
        return *this;
    }

    auto operator-=(const HField& other) -> HField& {
        for (size_t Comp = 0; Comp < NumComp; Comp++) {
            this->_Mat[Comp] -= other._Mat[Comp];
        }
        return *this;
    }

    auto operator*=(const double& other) -> HField& {
        for (size_t Comp = 0; Comp < NumComp; Comp++) {
            this->_Mat[Comp] *= other;
        }
        return *this;
    }

    friend auto operator+(HField lhs, const HField& rhs) -> HField {
        lhs += rhs;
        return lhs;
    }

    friend auto operator-(HField lhs, const HField& rhs) -> HField {
        lhs -= rhs;
        return lhs;
    }

    friend auto operator*(double lhs, HField rhs) -> HField {
        rhs *= lhs;
        return rhs;
    }

    [[nodiscard]] auto trace() const -> T;
    [[nodiscard]] auto dot() const -> T;

    void print();
};  // struct HField

template <RealValue T>
inline auto HField<T>::trace() const -> T {
    return _Mat[0] + _Mat[4] + _Mat[7] + _Mat[9];
}

template <RealValue T>
inline auto HField<T>::dot() const -> T {
    T Res = 0.0;
    for (const auto& Elem : _Mat) {
        Res += Elem * Elem;
    }
    return Res;
}

template <RealValue T>
inline void HField<T>::print() {
    std::cout << std::scientific;
    for (int Mu = 0; Mu < 4; Mu++) {
        for (int Nu = 0; Nu < 4; Nu++) {
            std::cout << std::format("{:>+12.2e}", (*this)(Mu, Nu));
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

template <RealValue T>
inline void reset(HField<T>& val) {
    std::fill(val._Mat.begin(), val._Mat.end(), 0.0);
}

/*--------------------------------------------------------------------------------------------------
    Hdf5 Types for core DataTypes
--------------------------------------------------------------------------------------------------*/

template <>
auto make_H5_Type<HField<RealF>>() {
    std::array<hsize_t, 1> Dims = {HField<RealF>::NumComp};
    hid_t                  DataTypeHid = H5Tarray_create(H5T_NATIVE_FLOAT, 1, Dims.data());
    return DataTypeHid;
}

template <>
auto make_H5_Type<HField<RealD>>() {
    std::array<hsize_t, 1> Dims = {HField<RealF>::NumComp};
    hid_t                  DataTypeHid = H5Tarray_create(H5T_NATIVE_DOUBLE, 1, Dims.data());
    return DataTypeHid;
}

/*--------------------------------------------------------------------------------------------------
    Math methods for Hfield objects
--------------------------------------------------------------------------------------------------*/

namespace HField_math {

// compute res = c * (A - B) element wise
template <RealValue T>
inline void diff(HField<T>& res, const HField<T>& lhs, const HField<T>& rhs, const double Offset = 1.0) {
    for (size_t Comp = 0; Comp < HField<T>::NumComp; Comp++) {
        res[Comp] = Offset * (lhs[Comp] - rhs[Comp]);
    }
}

// compute res = c * (A + B) element wise
template <RealValue T>
inline void sum(HField<T>& res, const HField<T>& lhs, const HField<T>& rhs, const double Offset = 1.0) {
    for (size_t Comp = 0; Comp < HField<T>::NumComp; Comp++) {
        res[Comp] = Offset * (lhs[Comp] + rhs[Comp]);
    }
}

// compute res = a * B element-wise
template <RealValue T>
inline void prod(HField<T>& res, const T lhs, const HField<T>& rhs) {
    for (size_t Comp = 0; Comp < HField<T>::NumComp; Comp++) {
        res[Comp] = lhs * rhs[Comp];
    }
}

}  // namespace HField_math
}  // namespace reticolo
