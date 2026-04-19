/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: lattice/indexing.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <climits>  // IWYU pragma: keep
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include "reticolo/core/types/coord.hpp"

namespace reticolo {

class Indexing {
  public:
    /* Types */
#if defined(SMALL_LATTICE)
    using size_type = unsigned short;
    constexpr static unsigned short MaxSize = USHRT_MAX;
#else
    using size_type = size_t;
    constexpr static size_t MaxSize = std::numeric_limits<size_type>::max();
#endif

    /* Lattice info */
    std::vector<size_type> Sizes;    // Lattice size in each dimension
    std::vector<size_type> SubVols;  // Subvolumes
    size_type              NSites;   // Total number of lattice sites
    size_type              Dims;     // Number of dimensions of the lattice

    /* Lattices of neighbour */
    std::vector<size_type> Next;
    std::vector<size_type> Prev;

    /* Constructor */
    Indexing(const std::vector<size_type>& shape);

    /* Get next/prev indexes */
    auto nextId(size_type site, size_type dir) -> size_type { return Next[(site * Dims) + dir]; }
    auto prevId(size_type site, size_type dir) -> size_type { return Prev[(site * Dims) + dir]; }
};

/* Constructor */
Indexing::Indexing(const std::vector<size_type>& shape)
    : Sizes(shape),
      SubVols(shape.size()),
      NSites(std::accumulate(Sizes.begin(), Sizes.end(), 1, std::multiplies<>())),
      Dims(shape.size()) {
    /* Fil the sub-volume vector */
    for (size_type SubDim = 0; SubDim < Dims - 1; SubDim++) {
        SubVols[SubDim] = std::accumulate(Sizes.begin() + SubDim + 1, Sizes.end(), 1, std::multiplies<>());
    }
    SubVols.back() = 1;
    /* support variables to keep track of position in the lattice */
    std::vector<size_type> Coord(Dims, 0);
    std::vector<size_type> PrevCoord(Dims);
    std::vector<size_type> NextCoord(Dims);
    /* Resize the vectors of neighbours */
    Next.resize(NSites * Dims, 0);
    Prev.resize(NSites * Dims, 0);
    /* Loop throught the lattice to generate all the closest neighbours */
    for (size_type Site = 0; Site < NSites; Site++) {
        for (size_type Dir = 0; Dir < Dims; Dir++) {
            NextCoord = Coord;
            NextCoord[Dir] = (Coord[Dir] + 1) % Sizes[Dir];
            Next[(Site * Dims) + Dir] = dot(NextCoord, SubVols);
            PrevCoord = Coord;
            PrevCoord[Dir] = (Coord[Dir] + (Sizes[Dir] - 1)) % Sizes[Dir];
            Prev[(Site * Dims) + Dir] = dot(PrevCoord, SubVols);
        }
        advance_coord(Sizes, Coord);
    }
};

}  // namespace reticolo
