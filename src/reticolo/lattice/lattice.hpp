/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: lattice/lattice.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "reticolo/types/core.hpp"

namespace reticolo {

template <typename T_field, size_t dim>
class Lattice {
    uintvect<dim> _N;        // vect storing the lattice size in N-dimensions
    uintvect<dim> _SubVols;  // Vector of subvolumes
    uint          _NSites;   // Total number of lattice sites

    std::vector<uintvect<dim>> _Next;
    std::vector<uintvect<dim>> _Prev;

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

        uintvect<dim> Coord;
        uintvect<dim> PrevCoord;
        uintvect<dim> NextCoord;

        std::fill(Coord.begin(), Coord.end(), 0);

        _Next.resize(_NSites);
        _Prev.resize(_NSites);

        for (uint Site = 0; Site < _NSites; advance_coord(_N, Coord), Site++) {
            for (uint Dir = 0; Dir < dim; Dir++) {
                NextCoord = Coord;
                NextCoord[Dir] = (Coord[Dir] + 1) % _N[Dir];
                _Next[Site][Dir] = site(NextCoord);
                PrevCoord = Coord;
                PrevCoord[Dir] = (Coord[Dir] + (_N[Dir] - 1)) % _N[Dir];
                _Prev[Site][Dir] = site(PrevCoord);
            }
        }

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
    [[nodiscard]] auto getSubVols() const -> uintvect<dim> { return _SubVols; }
    [[nodiscard]] auto getSizes() const -> uintvect<dim> { return _N; }

    // Get lattice index from 4d coordinates
    auto site(uintvect<dim>& coord) -> uint { return coord * _SubVols; };
    auto site(const uintvect<dim>& coord) const -> uint { return coord * _SubVols; };

    auto next(uint site, uint dir) -> T_field& { return _Field[_Next[site][dir]]; }
    auto next(uint site, uint dir) const -> const T_field& { return _Field[_Next[site][dir]]; }

    auto next(uintvect<dim> coord, uint dir) -> T_field& { return _Field[_Next[site(coord)][dir]]; }
    auto next(uintvect<dim> coord, uint dir) const -> const T_field& { return _Field[_Next[site(coord)][dir]]; }

    auto prev(uint site, uint dir) -> T_field& { return _Field[_Prev[site][dir]]; }
    auto prev(uint site, uint dir) const -> const T_field& { return _Field[_Prev[site][dir]]; }

    auto prev(uintvect<dim> coord, uint dir) -> T_field& { return _Field[_Prev[site(coord)][dir]]; }
    auto prev(uintvect<dim> coord, uint dir) const -> const T_field& { return _Field[_Prev[site(coord)][dir]]; }

    [[nodiscard]] auto memoryReport() const -> size_t {
        size_t Tot = 0;
        Tot += sizeof(T_field) * _Field.size();    // byte size of the main lattice
        Tot += sizeof(uint) * dim * _Prev.size();  // byte size of the prev index lattice
        Tot += sizeof(uint) * dim * _Next.size();  // byte size of the next index lattice
        Tot += sizeof(uint) * dim;                 // byte size of the lattice sizes vector
        Tot += sizeof(uint) * dim;                 // byte size of the subvolume vector
        Tot += sizeof(uint);                       // byte size of the number of lattice sizes

        return sizeof(T_field) * _Field.size();
    }
};

}  // namespace reticolo