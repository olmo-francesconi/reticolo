/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: lattice/lattice.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <H5public.h>

#include <concepts>
#include <cstdlib>
#include <memory>
#include <vector>

#include "reticolo/lattice/Indexing.hpp"
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

    /* Constructors */
    Lattice();                                                 // default
    Lattice(const std::vector<size_type>& shape);              // shape constructor
    Lattice(const std::vector<size_type>& shape, TField val);  // value-initialized
    Lattice(const Lattice& obj);                               // copy constructor
    template <typename T>                                      //
        requires(!std::same_as<TField, T>)                     // copy constructor from latttie of different type
    Lattice(const Lattice<T>& obj);                            //

    /* Destructor */
    ~Lattice();

    /* Copy assignment */
    auto operator=(const Lattice& other) -> Lattice&;

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
