/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/types/hfield.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <array>
#include <cstddef>
#include <iostream>

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

const size_t HfieldNComp = 10;

namespace reticolo {
// Basic matrix class to store h fields
struct HField {
  std::array<double, HfieldNComp> mat;

  HField() = default; // std::cout << "HField::HField() constructor called" <<
                      // std::endl; };
  ~HField(){};        // std::cout << "HField::~HField() destructor called" <<
                      // std::endl; };

  // Array-like access operators
  auto operator[](size_t index) -> double & { return mat[index]; };
  auto operator[](size_t index) const -> const double & { return mat[index]; };
  auto operator()(size_t mu, size_t nu) -> double & {
    if (nu < mu) {
      std::swap(mu, nu);
    }
    return mat[4 * mu + nu - (mu * (mu + 1)) / 2];
  }
  auto operator()(size_t mu, size_t nu) const -> const double & {
    if (nu < mu) {
      std::swap(mu, nu);
    }
    return mat[4 * mu + nu - (mu * (mu + 1)) / 2];
  }

  auto operator=(const HField &other) -> HField & {
    std::copy(other.mat.begin(), other.mat.end(), mat.begin());
    return *this;
  }

  [[nodiscard]] auto trace() const -> double;

  void print();
}; // struct HField

inline auto HField::trace() const -> double {
  return mat[0] + mat[4] + mat[7] + mat[9];
}

inline void HField::print() {
  std::cout << std::scientific;
  for (int mu = 0; mu < 4; mu++) {
    for (int nu = 0; nu < 4; nu++) {
      std::cout << (*this)(mu, nu) << "\t";
    }
    std::cout << '\n';
  }
  std::cout << '\n';
}

// Math methods for Hfield objects
namespace HField_math {
// compute res = c * (A - B) element wise
inline void diff(HField &res, const HField &A, const HField &B,
                 const double c = 1.0) {
  for (size_t Comp = 0; Comp < HfieldNComp; Comp++) {
    res[Comp] = c * (A[Comp] - B[Comp]);
  }
}

// compute res = c * (A + B) element wise
inline void sum(HField &res, const HField &A, const HField &B,
                const double c = 1.0) {
  for (size_t Comp = 0; Comp < HfieldNComp; Comp++) {
    res[Comp] = c * (A[Comp] + B[Comp]);
  }
}

inline void prod(HField &res, const double lhs, const HField &rhs) {
  for (size_t Comp = 0; Comp < HfieldNComp; Comp++) {
    res[Comp] = lhs * rhs[Comp];
  }
}

} // namespace HField_math
} // namespace reticolo