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
        vect<dim> N;  // vect storing the lattice size in N-dimensions
        uint n_sites; // Total number of lattice sites
        vect<dim> sub_vols;

        std::vector<T_field> field; // actual lattice
    public:
        // default constructor (does nothing)
        lattice(){};

        // Initialise a four-dimensional lattice of dimensions vect4({t,  x,  y,  z})
        lattice(vect<dim> sizes) { init(sizes); }

        // Initialise a four-dimensional lattice of dimensions vect4({t,  x,  y,  z})
        void init(vect<dim> sizes)
        {
            N = sizes;

            n_sites = std::accumulate(N.begin(), N.end(), 1, std::multiplies<uint>());

            for (int i = 0; i < dim - 1; i++)
                sub_vols[i] = std::accumulate(N.begin() + i + 1, N.end(), 1, std::multiplies<uint>());
            sub_vols.back() = 1;

            field.clear();
            field.resize(n_sites);
        };

        T_field &operator[](const uint site) { return field[site]; }
        const T_field &operator[](const uint site) const { return field[site]; }

        T_field &operator[](const vect<dim> &coord) { return field[site(coord)]; }
        const T_field &operator[](const vect<dim> &coord) const { return field[site(coord)]; }

        T_field &operator()(vect<dim> &coord) { return field[site(coord)]; }
        const T_field &operator()(const vect<dim> &coord) const { return field[site(coord)]; }

        uint getNt() const { return N[_t]; }
        uint getNx() const { return N[_x]; }
        uint getNy() const { return N[_y]; }
        uint getNz() const { return N[_z]; }
        uint getNsites() const { return n_sites; }
        uint getVolume() const { return n_sites; }
        vect<dim> getSizes() const { return N; }

        size_t memoryReport() const { return sizeof(T_field) * field.size(); }

        // Get lattice index from 4d coordinates
        uint site(vect<dim> &c) { return c * sub_vols; };
        const uint site(const vect<dim> &c) const { return c * sub_vols; };

        template <typename T, size_t d>
        friend uint next(const lattice<T, d> &l, const vect<d> &coord, const uint dir);
        template <typename T, size_t d>
        friend uint prev(const lattice<T, d> &l, const vect<d> &coord, const uint dir);
    };

    template <typename T, size_t dim>
    uint next(const lattice<T, dim> &l, const vect<dim> &coord, const uint dir)
    {
        vect<dim> next_coord = coord;
        next_coord[dir] = (next_coord[dir] + 1) % l.N[dir];
        return l.site(next_coord);
    }

    template <typename T, size_t dim>
    uint prev(const lattice<T, dim> &l, const vect<dim> &coord, const uint dir)
    {
        vect<dim> prev_coord = coord;
        prev_coord[dir] = (prev_coord[dir] - 1 + l.N[dir]) % l.N[dir];
        return l.site(prev_coord);
    }
} // namespace reticolo