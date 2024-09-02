/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: lattice/Indexing.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core_math.hpp"

namespace reticolo {

class Indexing {
  public:
    using SizeType = unsigned int;

    /* Lattice info */
    std::vector<SizeType> Sizes;    // Lattice size in each dimension
    std::vector<SizeType> SubVols;  // Subvolumes
    SizeType              NSites;   // Total number of lattice sites
    SizeType              Dims;     // Number of dimensions of the lattice

    /* Lattices of neighbour */
    std::vector<SizeType> Next;
    std::vector<SizeType> Prev;

    /* Constructor */
    Indexing(const std::vector<SizeType>& shape)
        : Sizes(shape),
          SubVols(shape.size()),
          NSites(std::accumulate(Sizes.begin(), Sizes.end(), 1, std::multiplies<>())),
          Dims(shape.size()) {
        /* Fil the sub-volume vector */
        for (SizeType SubDim = 0; SubDim < Dims - 1; SubDim++) {
            SubVols[SubDim] = std::accumulate(Sizes.begin() + SubDim + 1, Sizes.end(), 1, std::multiplies<>());
        }
        SubVols.back() = 1;

        /* support variables to keep track of position in the lattice */
        std::vector<SizeType> Coord(Dims, 0);
        std::vector<SizeType> PrevCoord(Dims);
        std::vector<SizeType> NextCoord(Dims);

        /* Resize the vectors of neighbours */
        Next.resize(NSites * Dims, 0);
        Prev.resize(NSites * Dims, 0);

        /* Loop throught the lattice to generate all the closest neighbours */
        for (SizeType Site = 0; Site < NSites; Site++) {
            for (SizeType Dir = 0; Dir < Dims; Dir++) {
                NextCoord = Coord;
                NextCoord[Dir] = (Coord[Dir] + 1) % Sizes[Dir];
                Next[Site * Dims + Dir] = dot(NextCoord, SubVols);
                PrevCoord = Coord;
                PrevCoord[Dir] = (Coord[Dir] + (Sizes[Dir] - 1)) % Sizes[Dir];
                Prev[Site * Dims + Dir] = dot(PrevCoord, SubVols);
            }
            advance_coord(Sizes, Coord);
        }
    };

    auto nextId(SizeType site, SizeType dir) -> SizeType { return Next[site + Dims + dir]; }
    auto prevId(SizeType site, SizeType dir) -> SizeType { return Prev[site + Dims + dir]; }
};

}  // namespace reticolo
