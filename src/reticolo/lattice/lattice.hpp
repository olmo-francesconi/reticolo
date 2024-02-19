/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: lattice/lattice.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <vector>

#include "reticolo/tools/types/basic.hpp"
#include "reticolo/tools/types/vect.hpp"

namespace reticolo {

template <typename T_field, size_t dim>
class lattice {
  uintvect<dim> N;  // vect storing the lattice size in N-dimensions
  uint n_sites;     // Total number of lattice sites
  uintvect<dim> sub_vols;

  std::vector<T_field> field;  // actual lattice

 public:
  // default constructor (does nothing)
  lattice(){};

  // constructor wrapper of init()
  lattice(uintvect<dim> sizes) { init(sizes); }

  // Initialise a four-dimensional lattice of dimensions vect4({t,  x,  y,  z})
  inline void init(uintvect<dim> sizes) {
    N = sizes;

    n_sites = std::accumulate(N.begin(), N.end(), 1, std::multiplies<>());

    for (int SubDim = 0; SubDim < dim - 1; SubDim++) {
      sub_vols[SubDim] = std::accumulate(N.begin() + SubDim + 1, N.end(), 1, std::multiplies<>());
    }
    sub_vols.back() = 1;

    field.clear();
    field.resize(n_sites);
  };

  auto operator[](const uint site) -> T_field& { return field[site]; }
  auto operator[](const uint site) const -> const T_field& { return field[site]; }

  auto operator[](const uintvect<dim>& coord) -> T_field& { return field[site(coord)]; }
  auto operator[](const uintvect<dim>& coord) const -> const T_field& { return field[site(coord)]; }

  auto operator()(uintvect<dim>& coord) -> T_field& { return field[site(coord)]; }
  auto operator()(const uintvect<dim>& coord) const -> const T_field& { return field[site(coord)]; }

  [[nodiscard]] auto getNt() const -> uint { return N[_t]; }
  [[nodiscard]] auto getNx() const -> uint { return N[_x]; }
  [[nodiscard]] auto getNy() const -> uint { return N[_y]; }
  [[nodiscard]] auto getNz() const -> uint { return N[_z]; }
  [[nodiscard]] auto getNsites() const -> uint { return n_sites; }
  [[nodiscard]] auto getVolume() const -> uint { return n_sites; }
  [[nodiscard]] auto getSizes() const -> uintvect<dim> { return N; }

  [[nodiscard]] auto memoryReport() const -> size_t { return sizeof(T_field) * field.size(); }

  // Get lattice index from 4d coordinates
  auto site(uintvect<dim>& coord) -> uint { return coord * sub_vols; };
  auto site(const uintvect<dim>& coord) const -> uint { return coord * sub_vols; };

  template <typename T, size_t d>
  friend auto next(const lattice<T, d>& lat, const uintvect<d>& coord, uint dir) -> uint;
  template <typename T, size_t d>
  friend auto prev(const lattice<T, d>& lat, const uintvect<d>& coord, uint dir) -> uint;
};

template <typename T, size_t dim>
auto next(const lattice<T, dim>& lat, const uintvect<dim>& coord, const uint dir) -> uint {
  uintvect<dim> NextCoord = coord;
  NextCoord[dir] = (NextCoord[dir] + 1) % lat.N[dir];
  return lat.site(NextCoord);
};

template <typename T, size_t dim>
auto prev(const lattice<T, dim>& lat, const uintvect<dim>& coord, const uint dir) -> uint {
  uintvect<dim> PrevCoord = coord;
  PrevCoord[dir] = (PrevCoord[dir] - 1 + lat.N[dir]) % lat.N[dir];
  return lat.site(PrevCoord);
};

}  // namespace reticolo