/*******************************************************************************

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

#include "reticolo/lattice/Indexing.hpp"
#include "reticolo/lattice/lattice.hpp"
#include "reticolo/tools/io_utils.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep

namespace reticolo {

/* Default constructor*/
template <typename TField>
Lattice<TField>::Lattice() = default;
/* Constructor */
template <typename TField>
Lattice<TField>::Lattice(const std::vector<size_type>& shape) {
    Idx = std::make_shared<Indexing>(shape);
    this->resize(Idx->NSites);
};
/* Value-initialized Constructor*/
template <typename TField>
Lattice<TField>::Lattice(const std::vector<size_type>& shape, TField val) {
    Idx = std::make_shared<Indexing>(shape);
    this->resize(Idx->NSites, val);
};
/* Copy Constructor */
template <typename TField>
Lattice<TField>::Lattice(const Lattice& obj) {
    Idx = obj.Idx;
    this->insert(this->begin(), obj.begin(), obj.end());
};
/* Copy Constructor (templated) */
template <typename TField>
template <typename T>
    requires(!std::same_as<TField, T>)
Lattice<TField>::Lattice(const Lattice<T>& obj) {
    Idx = obj.Idx;
    this->resize(Idx->NSites);
};
/* Destructor */
template <typename TField>
Lattice<TField>::~Lattice() = default;

/* Copy assignment */
template <typename TField>
auto Lattice<TField>::operator=(const Lattice& other) -> Lattice& {
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

}  // namespace reticolo
