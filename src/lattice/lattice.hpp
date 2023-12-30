#pragma once

#include "tools/types.hpp"

#include <vector>
#include <array>
#include <random>
#include <complex>

namespace LLR
{
    // namespace lattice
    // {
    template <typename T_field>
    class lattice
    {
        vect4 N;      // vect4 storing the lattice size in the four dimensions
        uint n_sites; // Total number of lattice sites

        std::vector<T_field> field; // actual lattice
    public:
        // default constructor (does nothing)
        lattice(){};

        // Initialise a four-dimensional lattice of dimensions vect4({t,  x,  y,  z})
        lattice(vect4 sizes) { init(sizes); }

        // Initialise a four-dimensional lattice of dimensions vect4({t,  x,  y,  z})
        void init(vect4 sizes)
        {
            N = sizes;

            n_sites = N[_t] * N[_x] * N[_y] * N[_z];

            field.clear();
            field.resize(n_sites);
        };

        T_field &operator[](const uint site) { return field[site]; }
        const T_field &operator[](const uint site) const { return field[site]; }

        T_field &operator[](const vect4 &coord) { return field[site(coord)]; }
        const T_field &operator[](const vect4 &coord) const { return field[site(coord)]; }

        T_field &operator()(vect4 &coord) { return field[site(coord)]; }
        const T_field &operator()(const vect4 &coord) const { return field[site(coord)]; }

        uint getNt() const { return N[_t]; }
        uint getNx() const { return N[_x]; }
        uint getNy() const { return N[_y]; }
        uint getNz() const { return N[_z]; }
        uint getNsites() const { return n_sites; }
        vect4 getSizes() const { return N; }

        size_t memoryReport() const { return sizeof(T_field) * field.size(); }

        // Get lattice index from 4d coordinates
        uint site(vect4 &c) { return c[_t] * N[_x] * N[_y] * N[_z] + c[_x] * N[_y] * N[_z] + c[_y] * N[_z] + c[_z]; };
        const uint site(const vect4 &c) const { return c[_t] * N[_x] * N[_y] * N[_z] + c[_x] * N[_y] * N[_z] + c[_y] * N[_z] + c[_z]; };

        // get index of next sites from current coordinates
        template <typename T>
        friend uint nt(const lattice<T> &l, const vect4 &coord);
        template <typename T>
        friend uint nx(const lattice<T> &l, const vect4 &coord);
        template <typename T>
        friend uint ny(const lattice<T> &l, const vect4 &coord);
        template <typename T>
        friend uint nz(const lattice<T> &l, const vect4 &coord);

        // get index of previous sites from current coordinates
        template <typename T>
        friend uint pt(const lattice<T> &l, const vect4 &coord);
        template <typename T>
        friend uint px(const lattice<T> &l, const vect4 &coord);
        template <typename T>
        friend uint py(const lattice<T> &l, const vect4 &coord);
        template <typename T>
        friend uint pz(const lattice<T> &l, const vect4 &coord);
    };

    template <typename T>
    uint nt(const lattice<T> &l, const vect4 &coord) { return l.site({(coord[_t] + 1) % l.N[_t], coord[_x], coord[_y], coord[_z]}); };
    template <typename T>
    uint nx(const lattice<T> &l, const vect4 &coord) { return l.site({coord[_t], (coord[_x] + 1) % l.N[_x], coord[_y], coord[_z]}); };
    template <typename T>
    uint ny(const lattice<T> &l, const vect4 &coord) { return l.site({coord[_t], coord[_x], (coord[_y] + 1) % l.N[_y], coord[_z]}); };
    template <typename T>
    uint nz(const lattice<T> &l, const vect4 &coord) { return l.site({coord[_t], coord[_x], coord[_y], (coord[_z] + 1) % l.N[_z]}); };

    template <typename T>
    uint pt(const lattice<T> &l, const vect4 &coord) { return l.site({(coord[_t] - 1 + l.N[_t]) % l.N[_t], coord[_x], coord[_y], coord[_z]}); };
    template <typename T>
    uint px(const lattice<T> &l, const vect4 &coord) { return l.site({coord[_t], (coord[_x] - 1 + l.N[_x]) % l.N[_x], coord[_y], coord[_z]}); };
    template <typename T>
    uint py(const lattice<T> &l, const vect4 &coord) { return l.site({coord[_t], coord[_x], (coord[_y] - 1 + l.N[_y]) % l.N[_y], coord[_z]}); };
    template <typename T>
    uint pz(const lattice<T> &l, const vect4 &coord) { return l.site({coord[_t], coord[_x], coord[_y], (coord[_z] - 1 + l.N[_z]) % l.N[_z]}); };
    // }
} // namespace LLR