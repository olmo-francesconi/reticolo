/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: lattice/Indexing.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <numeric>
#include <vector>

#include "reticolo/tools/io_utils.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"
#include "reticolo/types/core_math.hpp"

namespace reticolo {

class Indexing {
  protected:
    /* Lattice info */
    std::vector<int> m_Sizes;    // Lattice size in each dimension
    std::vector<int> m_SubVols;  // Subvolumes
    int              m_NSites;   // Total number of lattice sites
    int              m_Dims;     // Number of dimensions of the lattice

    /* Lattices of neighbour */
    std::vector<std::vector<int>> m_Next;
    std::vector<std::vector<int>> m_Prev;

  public:
    /* Constructor */
    Indexing(const std::vector<int>& sizes)
        : m_Sizes(sizes),
          m_SubVols(sizes.size()),
          m_NSites(std::accumulate(m_Sizes.begin(), m_Sizes.end(), 1, std::multiplies<>())),
          m_Dims(sizes.size()) {
        /* Fil the sub-volume vector */
        for (uint SubDim = 0; SubDim < m_Dims - 1; SubDim++) {
            m_SubVols[SubDim] = std::accumulate(m_Sizes.begin() + SubDim + 1, m_Sizes.end(), 1, std::multiplies<>());
        }
        m_SubVols.back() = 1;

        /* support variables to keep track of position in the lattice */
        std::vector<int> Coord(m_Dims, 0);
        std::vector<int> PrevCoord(m_Dims);
        std::vector<int> NextCoord(m_Dims);

        /* Resize the vectors of neighbours */
        m_Next.resize(m_NSites, std::vector<int>(m_Dims, 0));
        m_Prev.resize(m_NSites, std::vector<int>(m_Dims, 0));

        /* Loop throught the lattice to generate all the closest neighbours */

        for (int Site = 0; Site < m_NSites; Site++) {
            for (uint Dir = 0; Dir < m_Dims; Dir++) {
                NextCoord = Coord;
                NextCoord[Dir] = (Coord[Dir] + 1) % m_Sizes[Dir];
                m_Next[Site][Dir] = dot(NextCoord, m_SubVols);
                PrevCoord = Coord;
                PrevCoord[Dir] = (Coord[Dir] + (m_Sizes[Dir] - 1)) % m_Sizes[Dir];
                m_Prev[Site][Dir] = dot(PrevCoord, m_SubVols);
            }
            advance_coord(m_Sizes, Coord);
        }
    };
};

}  // namespace reticolo
