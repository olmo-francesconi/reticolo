/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: lattice/Indexing.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <vector>

namespace reticolo {

class Indexing {
  public:
    /* Types */
    using size_type = unsigned int;

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
    auto nextId(size_type site, size_type dir) -> size_type { return Next[site * Dims + dir]; }
    auto prevId(size_type site, size_type dir) -> size_type { return Prev[site * Dims + dir]; }
};

}  // namespace reticolo
