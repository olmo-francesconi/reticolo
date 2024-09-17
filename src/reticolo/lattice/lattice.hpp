/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: lattice/lattice.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <H5public.h>

#include <algorithm>
#include <concepts>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "reticolo/lattice/indexing.hpp"
#include "reticolo/tools/io_utils.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"

namespace reticolo {

template <typename TField>
class Lattice : public std::vector<TField> {
  public:
    /* Types */
    using size_type = Indexing::size_type;

    /* Public member variables */
    std::shared_ptr<Indexing> Idx;

    /* Default constructor */
    Lattice() = default;
    /* Shape constructor */
    Lattice(const std::vector<size_type>& shape) {
        Idx = std::make_shared<Indexing>(shape);
        this->resize(Idx->NSites);
    };
    /* Value-initialized contructor */
    Lattice(const std::vector<size_type>& shape, TField val) {
        Idx = std::make_shared<Indexing>(shape);
        this->resize(Idx->NSites, val);
    };
    /* Copy constructor */
    Lattice(const Lattice& obj) {
        Idx = obj.Idx;
        this->insert(this->begin(), obj.begin(), obj.end());
    };
    /* Copy constructor (templated) */
    template <typename T>
        requires(!std::same_as<TField, T>)
    Lattice(const Lattice<T>& obj) {
        Idx = obj.Idx;
        this->resize(Idx->NSites);
    };

    /* Destructor */
    ~Lattice() = default;

    /* Copy assignment */
    auto operator=(const Lattice& other) -> Lattice& {
        // Guard self assignment
        if (this == &other) {
            return *this;
        }
        // Copy only if compatible
        if (this->getSizes() == other.getSizes()) {
            std::copy(other.begin(), other.end(), this->begin());
        } else {
            std::cerr << IO::LI_erro() + "reticolo::Lattice - Copy Assigment failed [incompatible lattice sizes]\n";
            exit(EXIT_FAILURE);
        }
        return *this;
    }

    /* Getters for lattice parameters */
    [[nodiscard]] auto getNt() const -> size_type { return Idx->Sizes[_t]; }
    [[nodiscard]] auto getNx() const -> size_type { return Idx->Sizes[_x]; }
    [[nodiscard]] auto getNy() const -> size_type { return Idx->Sizes[_y]; }
    [[nodiscard]] auto getNz() const -> size_type { return Idx->Sizes[_z]; }
    [[nodiscard]] auto getNi(size_type Dir) const -> size_type { return Idx->Sizes[Dir]; }
    [[nodiscard]] auto getDim() const -> size_type { return Idx->Dims; }
    [[nodiscard]] auto getNsites() const -> size_type { return Idx->NSites; }
    [[nodiscard]] auto getSubVols() const -> std::vector<size_type> { return Idx->SubVols; }
    [[nodiscard]] auto getSizes() const -> std::vector<size_type> { return Idx->Sizes; }

    /* Access next and previous sites */
    [[nodiscard]] auto n(const size_type site, const size_type dir) -> TField& {
        return (*this)[Idx->nextId(site, dir)];
    }
    [[nodiscard]] auto n(const size_type site, const size_type dir) const -> const TField& {
        return (*this)[Idx->nextId(site, dir)];
    }
    [[nodiscard]] auto nn(size_type site, const std::vector<size_type>& dirs) -> TField& {
        for (const auto& Dir : dirs) {
            site = Idx->nextId(site, Dir);
        }
        return (*this)[site];
    }
    [[nodiscard]] auto nn(size_type site, const std::vector<size_type>& dirs) const -> const TField& {
        for (const auto& Dir : dirs) {
            site = Idx->nextId(site, Dir);
        }
        return (*this)[site];
    }

    [[nodiscard]] auto p(const size_type site, const size_type dir) -> TField& {
        return (*this)[Idx->prevId(site, dir)];
    }
    [[nodiscard]] auto p(const size_type site, const size_type dir) const -> const TField& {
        return (*this)[Idx->prevId(site, dir)];
    }
    [[nodiscard]] auto pp(size_type site, const std::vector<size_type>& dirs) -> TField& {
        for (const auto& Dir : dirs) {
            site = Idx->prevId(site, Dir);
        }
        return (*this)[site];
    }
    [[nodiscard]] auto pp(size_type site, const std::vector<size_type>& dirs) const -> const TField& {
        for (const auto& Dir : dirs) {
            site = Idx->prevId(site, Dir);
        }
        return (*this)[site];
    }
};

}  // namespace reticolo
