/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: lattice/lattice.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <H5public.h>
#include <yaml-cpp/node/node.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "reticolo/tools/io_utils.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"
#include "reticolo/types/core_math.hpp"

namespace reticolo {

// enum neighbour { prev = 0, next = 1 };

template <size_t dims>
class Indexing {
  public:
    /* Lattice info */
    intvect<dims> _Sizes;    // Lattice size in each dimension
    intvect<dims> _SubVols;  // Subvolumes
    int           _NSites;   // Total number of lattice sites
    /* Lattices of neighbour */
    std::vector<intvect<dims>> _Next;
    std::vector<intvect<dims>> _Prev;
    /* Constructor */
    Indexing(intvect<dims> sizes) : _Sizes(sizes) {
        _NSites = std::accumulate(_Sizes.begin(), _Sizes.end(), 1, std::multiplies<>());
        for (uint SubDim = 0; SubDim < dims - 1; SubDim++) {
            _SubVols[SubDim] = std::accumulate(_Sizes.begin() + SubDim + 1, _Sizes.end(), 1, std::multiplies<>());
        }
        _SubVols.back() = 1;
        intvect<dims> Coord;
        intvect<dims> PrevCoord;
        intvect<dims> NextCoord;
        std::fill(Coord.begin(), Coord.end(), 0);
        _Next.resize(_NSites);
        _Prev.resize(_NSites);
        for (int Site = 0; Site < _NSites; advance_coord(_Sizes, Coord), Site++) {
            for (uint Dir = 0; Dir < dims; Dir++) {
                NextCoord = Coord;
                NextCoord[Dir] = (Coord[Dir] + 1) % _Sizes[Dir];
                _Next[Site][Dir] = site(NextCoord);
                PrevCoord = Coord;
                PrevCoord[Dir] = (Coord[Dir] + (_Sizes[Dir] - 1)) % _Sizes[Dir];
                _Prev[Site][Dir] = site(PrevCoord);
            }
        }
    };

    void init(const YAML::Node& config) {
        _Sizes = config["sizes"];
        _NSites = std::accumulate(_Sizes.begin(), _Sizes.end(), 1, std::multiplies<>());
        for (uint SubDim = 0; SubDim < dims - 1; SubDim++) {
            _SubVols[SubDim] = std::accumulate(_Sizes.begin() + SubDim + 1, _Sizes.end(), 1, std::multiplies<>());
        }
        _SubVols.back() = 1;
        intvect<dims> Coord;
        intvect<dims> PrevCoord;
        intvect<dims> NextCoord;
        std::fill(Coord.begin(), Coord.end(), 0);
        _Next.resize(_NSites);
        _Prev.resize(_NSites);
        for (int Site = 0; Site < _NSites; advance_coord(_Sizes, Coord), Site++) {
            for (uint Dir = 0; Dir < dims; Dir++) {
                NextCoord = Coord;
                NextCoord[Dir] = (Coord[Dir] + 1) % _Sizes[Dir];
                _Next[Site][Dir] = site(NextCoord);
                PrevCoord = Coord;
                PrevCoord[Dir] = (Coord[Dir] + (_Sizes[Dir] - 1)) % _Sizes[Dir];
                _Prev[Site][Dir] = site(PrevCoord);
            }
        }
    };

    /* Get the lattice site from the coordinates*/
    auto site(intvect<dims>& coord) -> int { return dot(coord, _SubVols); };
    auto site(const intvect<dims>& coord) const -> int { return dot(coord, _SubVols); };
};

template <typename TField, size_t dim>
class Lattice : public Indexing<dim> {
    /* Vector string the Field components */
    std::vector<TField> _Field;

    /* Introduced names from Indexing<dim> (avoids using this->...) */
    using Indexing<dim>::_Sizes;
    using Indexing<dim>::_SubVols;
    using Indexing<dim>::_NSites;
    using Indexing<dim>::_Next;
    using Indexing<dim>::_Prev;

  public:
    /* Constructor */
    Lattice(intvect<dim> sizes) : Indexing<dim>(sizes) {
        _Field.clear();
        _Field.resize(_NSites);
    };

    void init(const YAML::Node& config) {
        Indexing<dim>::init(config);
        _Field.clear();
        _Field.resize(_NSites);
    };

    /* Copy assignment */
    auto operator=(const Lattice& other) -> Lattice& {
        // Guard self assignment
        if (this == &other) {
            return *this;
        }
        // Copy only if compatible
        if (_Sizes == other._Sizes) {
            std::copy(other._Field.begin(), other._Field.end(), _Field.begin());
        } else {
            std::cerr << IO::LI_erro() + "reticolo::Lattice - Copy Assigment failed [incompatible lattice sizes]\n";
            exit(EXIT_FAILURE);
        }
        return *this;
    }

    /* Data accessing operators */
    auto operator[](const int site) -> TField& { return _Field[site]; }
    auto operator[](const int site) const -> const TField& { return _Field[site]; }

    /* Getters for lattice parameters */
    [[nodiscard]] auto getNt() const -> int { return _Sizes[_t]; }
    [[nodiscard]] auto getNx() const -> int { return _Sizes[_x]; }
    [[nodiscard]] auto getNy() const -> int { return _Sizes[_y]; }
    [[nodiscard]] auto getNz() const -> int { return _Sizes[_z]; }
    [[nodiscard]] auto getNi(int Dir) const -> int { return _Sizes[Dir]; }
    [[nodiscard]] auto getDim() const -> int { return dim; }
    [[nodiscard]] auto getNsites() const -> int { return _NSites; }
    [[nodiscard]] auto getVolume() const -> int { return _NSites; }
    [[nodiscard]] auto getSubVols() const -> intvect<dim> { return _SubVols; }
    [[nodiscard]] auto getSizes() const -> intvect<dim> { return _Sizes; }

    /* Access next and previous sites */
    auto               next(int site, int dir) -> TField& { return _Field[_Next[site][dir]]; }
    auto               next(int site, int dir) const -> const TField& { return _Field[_Next[site][dir]]; }
    auto               nextId(int site, int dir) -> int& { return _Next[site][dir]; }
    [[nodiscard]] auto nextId(int site, int dir) const -> const int& { return _Next[site][dir]; }
    auto               prev(int site, int dir) -> TField& { return _Field[_Prev[site][dir]]; }
    auto               prev(int site, int dir) const -> const TField& { return _Field[_Prev[site][dir]]; }
    auto               prevId(int site, int dir) -> int& { return _Prev[site][dir]; }
    [[nodiscard]] auto prevId(int site, int dir) const -> const int& { return _Prev[site][dir]; }

    /* Save current field configuration */
    inline auto save_Configuration(const std::string& FilePath) -> hsize_t {
        hsize_t FileSize;
        // try {
        //     // Open file
        //     H5::H5File File(FilePath, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        //     // Create dataspace
        //     std::array<hsize_t, 1> Dims = {static_cast<hsize_t>(_NSites)};  // dim sizes of ds (on disk)
        //     H5::DataSpace          DataSpace(1, Dims.data());
        //     // Create datatype
        //     auto DataType = make_H5_Type<TField>();
        //     // Create dataset
        //     H5::DataSet Dataset = File.createDataSet("field", DataType, DataSpace);
        //     // Write dataset
        //     Dataset.write(_Field.data(), DataType);
        //     // Get file size
        //     FileSize = File.getFileSize();
        // } catch (H5::Exception& _) {
        //     exit(EXIT_FAILURE);
        // }
        return FileSize;
    }

    /* Read configuration from disk */
    inline auto read_Configuration(const std::string& FilePath) -> hsize_t {
        hsize_t DataSize;
        // try {
        //     // Open file
        //     H5::H5File File(FilePath, H5F_ACC_RDONLY, H5P_DEFAULT, H5P_DEFAULT);
        //     // Open DataSet
        //     H5::DataSet DataSet = File.openDataSet("field");
        //     // check DataTypes
        //     H5::DataType FileDataType = DataSet.getDataType();
        //     H5::DataType LatticeDataType = make_H5_Type<TField>();
        //     if (FileDataType != LatticeDataType) {
        //         H5::DataTypeIException Exeption("lattice::read_Configuration()", "File and data DataType differ");
        //         throw(Exeption);
        //     }
        //     // check DataSpaces
        //     H5::DataSpace FileDataSpace = DataSet.getSpace();
        //     hsize_t       FileDims = FileDataSpace.getSimpleExtentNdims();
        //     if (FileDims != _NSites) {
        //         H5::DataSpaceIException Exeption("lattice::read_Configuration()", "File and data DataSpaces differ");
        //         throw(Exeption);
        //     }
        //     // read data
        //     DataSet.read(_Field.data(), LatticeDataType);
        //     // get data size
        //     DataSize = DataSet.getInMemDataSize();
        // } catch (H5::DataTypeIException& e) {
        //     std::cerr << IO::LI_erro() << e.getFuncName() << " : " << e.getDetailMsg() << '\n';
        //     exit(EXIT_FAILURE);
        // } catch (H5::DataSpaceIException& e) {
        //     std::cerr << IO::LI_erro() << e.getFuncName() << " : " << e.getDetailMsg() << '\n';
        //     exit(EXIT_FAILURE);
        // }
        return DataSize;
    }
};

}  // namespace reticolo
