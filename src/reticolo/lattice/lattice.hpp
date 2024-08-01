/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: lattice/lattice.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <H5public.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "reticolo/lattice/Indexing.hpp"
#include "reticolo/tools/io_utils.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"

namespace reticolo {

template <typename TField>
class Lattice : public Indexing {
    /* Vector string the Field components */
    std::vector<TField> m_Field;

    /* Introduced names from Indexing<dim> (avoids using this->...) */
    using Indexing::m_Dims;
    using Indexing::m_Next;
    using Indexing::m_NSites;
    using Indexing::m_Prev;
    using Indexing::m_Sizes;
    using Indexing::m_SubVols;

  public:
    /* Constructor */
    Lattice(const std::vector<int>& sizes) : Indexing(sizes) {
        std::cout << "initializing the lattice data..\n";
        m_Field.clear();
        m_Field.resize(m_NSites);
        std::cout << "allocated" << IO::pretty_bytes(m_Field.size() * sizeof(TField)) << "\n";
    };

    /* Copy assignment */
    auto operator=(const Lattice& other) -> Lattice& {
        // Guard self assignment
        if (this == &other) {
            return *this;
        }
        // Copy only if compatible
        if (m_Sizes == other.m_Sizes) {
            std::copy(other.m_Field.begin(), other.m_Field.end(), m_Field.begin());
        } else {
            std::cerr << IO::LI_erro() + "reticolo::Lattice - Copy Assigment failed [incompatible lattice sizes]\n";
            exit(EXIT_FAILURE);
        }
        return *this;
    }

    /* Expose stl iterators */
    auto begin() -> std::vector<TField>::iterator { return m_Field.begin(); };
    auto end() -> std::vector<TField>::iterator { return m_Field.end(); };
    auto cbegin() -> std::vector<TField>::iterator { return m_Field.cbegin(); };
    auto cend() -> std::vector<TField>::iterator { return m_Field.cend(); };
    auto rbegin() -> std::vector<TField>::iterator { return m_Field.rbegin(); };
    auto rend() -> std::vector<TField>::iterator { return m_Field.rend(); };

    /* Data accessing operators */
    auto operator[](const int site) -> TField& { return m_Field[site]; }
    auto operator[](const int site) const -> const TField& { return m_Field[site]; }

    /* Getters for lattice parameters */
    [[nodiscard]] auto getNt() const -> int { return m_Sizes[_t]; }
    [[nodiscard]] auto getNx() const -> int { return m_Sizes[_x]; }
    [[nodiscard]] auto getNy() const -> int { return m_Sizes[_y]; }
    [[nodiscard]] auto getNz() const -> int { return m_Sizes[_z]; }
    [[nodiscard]] auto getNi(int Dir) const -> int { return m_Sizes[Dir]; }
    [[nodiscard]] auto getDim() const -> int { return m_Dims; }
    [[nodiscard]] auto getNsites() const -> int { return m_NSites; }
    [[nodiscard]] auto getVolume() const -> int { return m_NSites; }
    [[nodiscard]] auto getSubVols() const -> std::vector<int> { return m_SubVols; }
    [[nodiscard]] auto getSizes() const -> std::vector<int> { return m_Sizes; }

    /* Access next and previous sites */
    [[nodiscard]] auto next(int site, int dir) -> TField& { return m_Field[m_Next[site][dir]]; }
    [[nodiscard]] auto next(int site, int dir) const -> const TField& { return m_Field[m_Next[site][dir]]; }
    [[nodiscard]] auto nextId(int site, int dir) -> int& { return m_Next[site][dir]; }
    [[nodiscard]] auto nextId(int site, int dir) const -> const int& { return m_Next[site][dir]; }
    [[nodiscard]] auto prev(int site, int dir) -> TField& { return m_Field[m_Prev[site][dir]]; }
    [[nodiscard]] auto prev(int site, int dir) const -> const TField& { return m_Field[m_Prev[site][dir]]; }
    [[nodiscard]] auto prevId(int site, int dir) -> int& { return m_Prev[site][dir]; }
    [[nodiscard]] auto prevId(int site, int dir) const -> const int& { return m_Prev[site][dir]; }

    /* Save current field configuration */
    inline auto save_Configuration(const std::string& FilePath) -> hsize_t {
        hsize_t FileSize;
        return FileSize;
    }

    /* Read configuration from disk */
    inline auto read_Configuration(const std::string& FilePath) -> hsize_t {
        hsize_t DataSize;
        return DataSize;
    }
};

}  // namespace reticolo
