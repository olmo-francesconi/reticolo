/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: lattice/inst/indexing.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#include "reticolo/lattice/indexing.hpp"

#include <functional>
#include <numeric>
#include <vector>

#include "reticolo/types/core_math.hpp"

namespace reticolo {

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
            Next[Site * Dims + Dir] = dot(NextCoord, SubVols);
            PrevCoord = Coord;
            PrevCoord[Dir] = (Coord[Dir] + (Sizes[Dir] - 1)) % Sizes[Dir];
            Prev[Site * Dims + Dir] = dot(PrevCoord, SubVols);
        }
        advance_coord(Sizes, Coord);
    }
};

}  // namespace reticolo
