/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/types/vect.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <array>
#include <iostream>
#include <numeric>

namespace reticolo {
// vector types
template <size_t dim> using uintvect = std::array<uint, dim>;

template <size_t dim>
auto operator*(const uintvect<dim> &lhs, const uintvect<dim> &rhs) -> uint {
  uint Res = 0;
  for (int Comp = 0; Comp < dim; Comp++) {
    Res += lhs[Comp] * rhs[Comp];
  }
  return Res;
}

template <size_t dim>
inline void advance_coord(const uintvect<dim> &sizes, uintvect<dim> &coord) {
  coord.back()++;
  for (int Dir = dim - 1; Dir >= 0; Dir--) {
    if (coord[Dir] == sizes[Dir]) {
      coord[Dir] = 0;
      coord[Dir - 1]++;
    } else {
      return;
    }
  }
}

template <size_t dim>
inline auto get_volume(const uintvect<dim> &Sizes) -> uint {
  return std::accumulate(Sizes.begin(), Sizes.end(), 1, std::multiplies<>());
}

template <size_t dim> inline void print(const uintvect<dim> &Vect) {
  for (int Comp = 0; Comp < dim; Comp++) {
    std::cout << Vect[Comp];
  }
  std::cout << '\n';
}
} // namespace reticolo