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
#include <random>
#include <string>
#include <vector>

#include "reticolo/lattice/Indexing.hpp"
#include "reticolo/tools/io_utils.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"
#include "reticolo/types/core_math.hpp"
#include "reticolo/types/hfield.hpp"
#include "reticolo/types/random.hpp"

namespace reticolo {

template <typename TField>
class Lattice : public std::vector<TField> {
  public:
    std::shared_ptr<Indexing> Idx;

    using SizeType = Indexing::SizeType;

    /* Constructor */
    Lattice(const std::vector<SizeType>& shape) {
        Idx = std::make_shared<Indexing>(shape);
        this->resize(Idx->NSites);
    };
    /* Value-initialized Constructor*/
    Lattice(const std::vector<SizeType>& shape, TField val) {
        Idx = std::make_shared<Indexing>(shape);
        this->resize(Idx->NSites, val);
    };
    /* Copy Constructor */
    Lattice(const Lattice& obj) {
        Idx = obj.Idx;
        this->insert(this->begin(), obj.begin(), obj.end());
    };
    /* Copy Constructor (templated) */
    template <typename T>
        requires(!std::same_as<TField, T>)
    Lattice(const Lattice<T>& obj) {
        Idx = obj.Idx;
        this->resize(Idx->NSites);
        std::cout << this->size() << "\n";
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
    [[nodiscard]] auto getNt() const -> SizeType { return Idx->Sizes[_t]; }
    [[nodiscard]] auto getNx() const -> SizeType { return Idx->Sizes[_x]; }
    [[nodiscard]] auto getNy() const -> SizeType { return Idx->Sizes[_y]; }
    [[nodiscard]] auto getNz() const -> SizeType { return Idx->Sizes[_z]; }
    [[nodiscard]] auto getNi(SizeType Dir) const -> SizeType { return Idx->Sizes[Dir]; }
    [[nodiscard]] auto getDim() const -> SizeType { return Idx->Dims; }
    [[nodiscard]] auto getNsites() const -> SizeType { return Idx->NSites; }
    [[nodiscard]] auto getSubVols() const -> std::vector<SizeType> { return Idx->SubVols; }
    [[nodiscard]] auto getSizes() const -> std::vector<SizeType> { return Idx->Sizes; }

    /* Access next and previous sites */
    [[nodiscard]] auto n(const SizeType site, const SizeType dir) -> TField& { return (*this)[Idx->nextId(site, dir)]; }
    [[nodiscard]] auto n(const SizeType site, const SizeType dir) const -> const TField& {
        return (*this)[Idx->nextId(site, dir)];
    }
    [[nodiscard]] auto nn(SizeType site, const std::vector<SizeType>& dirs) -> TField& {
        for (const auto& Dir : dirs) {
            site = Idx->nextId(site, Dir);
        }
        return (*this)[site];
    }
    [[nodiscard]] auto nn(SizeType site, const std::vector<SizeType>& dirs) const -> const TField& {
        for (const auto& Dir : dirs) {
            site = Idx->nextId(site, Dir);
        }
        return (*this)[site];
    }

    [[nodiscard]] auto p(const SizeType site, const SizeType dir) -> TField& { return (*this)[Idx->prevId(site, dir)]; }
    [[nodiscard]] auto p(const SizeType site, const SizeType dir) const -> const TField& {
        return (*this)[Idx->prevId(site, dir)];
    }
    [[nodiscard]] auto pp(SizeType site, const std::vector<SizeType>& dirs) -> TField& {
        for (const auto& Dir : dirs) {
            site = Idx->prevId(site, Dir);
        }
        return (*this)[site];
    }
    [[nodiscard]] auto pp(SizeType site, const std::vector<SizeType>& dirs) const -> const TField& {
        for (const auto& Dir : dirs) {
            site = Idx->prevId(site, Dir);
        }
        return (*this)[site];
    }
};

/*--------------------------------------------------------------------------------------------------
  Lattice types definitions
--------------------------------------------------------------------------------------------------*/

using RealLatticeF = Lattice<RealF>;
using RealLatticeD = Lattice<RealD>;
using ComplexLatticeF = Lattice<ComplexF>;
using ComplexLatticeD = Lattice<ComplexD>;
using GRLatticeF = Lattice<HField<RealF>>;
using GRLatticeD = Lattice<HField<RealD>>;

}  // namespace reticolo
