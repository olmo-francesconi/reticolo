/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: lattice/lattice.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "reticolo/tools/types/core.hpp"

namespace reticolo {

template <typename T_field, size_t dim>
class Lattice {
    uintvect<dim> _N;        // vect storing the lattice size in N-dimensions
    uintvect<dim> _SubVols;  // Vector of subvolumes
    uint          _NSites;   // Total number of lattice sites

    std::vector<T_field> _Field;  // actual lattice

  public:
    // default constructor (does nothing)
    Lattice() = default;

    // constructor wrapper of init()
    Lattice(uintvect<dim> sizes) { init(sizes); }

    // Initialise a four-dimensional lattice of dimensions vect4({t,  x,  y,  z})
    inline void init(uintvect<dim> sizes) {
        _N = sizes;

        _NSites = std::accumulate(_N.begin(), _N.end(), 1, std::multiplies<>());

        for (int SubDim = 0; SubDim < dim - 1; SubDim++) {
            _SubVols[SubDim] = std::accumulate(_N.begin() + SubDim + 1, _N.end(), 1, std::multiplies<>());
        }
        _SubVols.back() = 1;

        _Field.clear();
        _Field.resize(_NSites);
    };

    auto operator[](const uint site) -> T_field& { return _Field[site]; }
    auto operator[](const uint site) const -> const T_field& { return _Field[site]; }

    auto operator[](const uintvect<dim>& coord) -> T_field& { return _Field[site(coord)]; }
    auto operator[](const uintvect<dim>& coord) const -> const T_field& { return _Field[site(coord)]; }

    auto operator()(uintvect<dim>& coord) -> T_field& { return _Field[site(coord)]; }
    auto operator()(const uintvect<dim>& coord) const -> const T_field& { return _Field[site(coord)]; }

    [[nodiscard]] auto getNt() const -> uint { return _N[_t]; }
    [[nodiscard]] auto getNx() const -> uint { return _N[_x]; }
    [[nodiscard]] auto getNy() const -> uint { return _N[_y]; }
    [[nodiscard]] auto getNz() const -> uint { return _N[_z]; }
    [[nodiscard]] auto getNsites() const -> uint { return _NSites; }
    [[nodiscard]] auto getVolume() const -> uint { return _NSites; }
    [[nodiscard]] auto getSizes() const -> uintvect<dim> { return _N; }

    [[nodiscard]] auto memoryReport() const -> size_t { return sizeof(T_field) * _Field.size(); }

    // Get lattice index from 4d coordinates
    auto site(uintvect<dim>& coord) -> uint { return coord * _SubVols; };
    auto site(const uintvect<dim>& coord) const -> uint { return coord * _SubVols; };

    template <typename T, size_t d>
    friend auto next(const Lattice<T, d>& lat, const uintvect<d>& coord, uint dir) -> uint;
    template <typename T, size_t d>
    friend auto prev(const Lattice<T, d>& lat, const uintvect<d>& coord, uint dir) -> uint;
};

template <typename T, size_t dim>
auto next(const Lattice<T, dim>& lat, const uintvect<dim>& coord, const uint dir) -> uint {
    uintvect<dim> NextCoord = coord;
    NextCoord[dir] = (NextCoord[dir] + 1) % lat._N[dir];
    return lat.site(NextCoord);
};

template <typename T, size_t dim>
auto prev(const Lattice<T, dim>& lat, const uintvect<dim>& coord, const uint dir) -> uint {
    uintvect<dim> PrevCoord = coord;
    PrevCoord[dir] = (PrevCoord[dir] - 1 + lat._N[dir]) % lat._N[dir];
    return lat.site(PrevCoord);
};

}  // namespace reticolo