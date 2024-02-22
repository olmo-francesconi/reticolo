/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/types/hfield.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <iostream>
#include <utility>

#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep

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

const size_t HfieldNumComp = 10;

namespace reticolo {
// Basic matrix class to store h fields
template <RealValue T>
class HField {
    std::array<T, HfieldNumComp> _Mat;

  public:
    HField() = default;
    HField(T val) { std::fill(_Mat.begin(), _Mat.end(), val); }
    HField(HField&& other) = default;

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

    [[nodiscard]] auto trace() const -> T;

    void print();
};  // struct HField

template <RealValue T>
inline auto HField<T>::trace() const -> T {
    return _Mat[0] + _Mat[4] + _Mat[7] + _Mat[9];
}

template <RealValue T>
inline void HField<T>::print() {
    std::cout << std::scientific;
    for (int Mu = 0; Mu < 4; Mu++) {
        for (int Nu = 0; Nu < 4; Nu++) {
            std::cout << (*this)(Mu, Nu) << "\t";
        }
        std::cout << '\n';
    }
    std::cout << '\n';
}

// Math methods for Hfield objects
namespace HField_math {
// compute res = c * (A - B) element wise
template <RealValue T>
inline void diff(HField<T>& res, const HField<T>& lhs, const HField<T>& rhs, const double Offset = 1.0) {
    for (size_t Comp = 0; Comp < HfieldNumComp; Comp++) {
        res[Comp] = Offset * (lhs[Comp] - rhs[Comp]);
    }
}

// compute res = c * (A + B) element wise
template <RealValue T>
inline void sum(HField<T>& res, const HField<T>& lhs, const HField<T>& rhs, const double Offset = 1.0) {
    for (size_t Comp = 0; Comp < HfieldNumComp; Comp++) {
        res[Comp] = Offset * (lhs[Comp] + rhs[Comp]);
    }
}

// compute res = a * B element-wise
template <RealValue T>
inline void prod(HField<T>& res, const T lhs, const HField<T>& rhs) {
    for (size_t Comp = 0; Comp < HfieldNumComp; Comp++) {
        res[Comp] = lhs * rhs[Comp];
    }
}

}  // namespace HField_math
}  // namespace reticolo