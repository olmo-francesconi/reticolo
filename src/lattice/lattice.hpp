/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: lattice/lattice.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include "tools/types.hpp"

#include <vector>
#include <array>
#include <random>
#include <complex>

namespace reticolo
{
    // namespace lattice
    // {
    template <typename T_field, size_t dim>
    class lattice
    {
        uintvect<dim> N;  // vect storing the lattice size in N-dimensions
        uint n_sites; // Total number of lattice sites
        uintvect<dim> sub_vols;

        std::vector<T_field> field; // actual lattice
    public:
        // default constructor (does nothing)
        lattice(){};

        // Initialise a four-dimensional lattice of dimensions vect4({t,  x,  y,  z})
        lattice(uintvect<dim> sizes) { init(sizes); }

        // Initialise a four-dimensional lattice of dimensions vect4({t,  x,  y,  z})
        void init(uintvect<dim> sizes);

        T_field &operator[](const uint site) { return field[site]; }
        const T_field &operator[](const uint site) const { return field[site]; }

        T_field &operator[](const uintvect<dim> &coord) { return field[site(coord)]; }
        const T_field &operator[](const uintvect<dim> &coord) const { return field[site(coord)]; }

        T_field &operator()(uintvect<dim> &coord) { return field[site(coord)]; }
        const T_field &operator()(const uintvect<dim> &coord) const { return field[site(coord)]; }

        uint getNt() const { return N[_t]; }
        uint getNx() const { return N[_x]; }
        uint getNy() const { return N[_y]; }
        uint getNz() const { return N[_z]; }
        uint getNsites() const { return n_sites; }
        uint getVolume() const { return n_sites; }
        uintvect<dim> getSizes() const { return N; }

        size_t memoryReport() const { return sizeof(T_field) * field.size(); }

        // Get lattice index from 4d coordinates
        uint site(uintvect<dim> &c) { return c * sub_vols; };
        const uint site(const uintvect<dim> &c) const { return c * sub_vols; };

        template <typename T, size_t d>
        friend uint next(const lattice<T, d> &l, const uintvect<d> &coord, const uint dir);
        template <typename T, size_t d>
        friend uint prev(const lattice<T, d> &l, const uintvect<d> &coord, const uint dir);
    };

    template <typename T, size_t dim>
    uint next(const lattice<T, dim> &l, const uintvect<dim> &coord, const uint dir)
    {
        uintvect<dim> next_coord = coord;
        next_coord[dir] = (next_coord[dir] + 1) % l.N[dir];
        return l.site(next_coord);
    };

    template <typename T, size_t dim>
    uint prev(const lattice<T, dim> &l, const uintvect<dim> &coord, const uint dir)
    {
        uintvect<dim> prev_coord = coord;
        prev_coord[dir] = (prev_coord[dir] - 1 + l.N[dir]) % l.N[dir];
        return l.site(prev_coord);
    };

} // namespace reticolo