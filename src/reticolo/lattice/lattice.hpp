/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: lattice/lattice.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

 ******************************************************************************/

#pragma once

#include <H5Cpp.h>

#include <array>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "reticolo/tools/io_utils.hpp"
#include "reticolo/types/core.hpp"

namespace reticolo {

template <size_t dim>
class Indexing {
  public:
    uintvect<dim> _Sizes;    // Lattice size in each dimension
    uintvect<dim> _SubVols;  // Subvolumes
    uint          _NSites;   // Total number of lattice sites

    std::vector<uintvect<dim>> _Next;
    std::vector<uintvect<dim>> _Prev;

    /* Constructor */
    Indexing(uintvect<dim> sizes) : _Sizes(sizes) {
        _NSites = std::accumulate(_Sizes.begin(), _Sizes.end(), 1, std::multiplies<>());

        for (uint SubDim = 0; SubDim < dim - 1; SubDim++) {
            _SubVols[SubDim] = std::accumulate(_Sizes.begin() + SubDim + 1, _Sizes.end(), 1, std::multiplies<>());
        }
        _SubVols.back() = 1;

        uintvect<dim> Coord;
        uintvect<dim> PrevCoord;
        uintvect<dim> NextCoord;
        std::fill(Coord.begin(), Coord.end(), 0);
        _Next.resize(_NSites);
        _Prev.resize(_NSites);
        for (uint Site = 0; Site < _NSites; advance_coord(_Sizes, Coord), Site++) {
            for (uint Dir = 0; Dir < dim; Dir++) {
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
    auto site(uintvect<dim>& coord) -> uint { return coord * _SubVols; };
    auto site(const uintvect<dim>& coord) const -> uint { return coord * _SubVols; };
};

template <typename TField, size_t dim>
class Lattice : public Indexing<dim> {
    // /* Indexing of the lattice*/
    // Indexing<dim> _Idx;

    /* Vector string the Field components */
    std::vector<TField> _Field;

  public:
    /* Constructor */
    Lattice(uintvect<dim> sizes) : Indexing<dim>(sizes) {
        _Field.clear();
        _Field.resize(this->_NSites);
    };

    /* Copy assignment */
    auto operator=(const Lattice& other) -> Lattice& {
        // Guard self assignment
        if (this == &other) {
            return *this;
        }
        // Copy only if compatible
        if (this->_Sizes == other._Sizes) {
            std::copy(other._Field.begin(), other._Field.end(), _Field.begin());
        } else {
            std::cerr << IO::LI_erro() + "reticolo::Lattice - Copy Assigment failed [incompatible lattice sizes]\n";
            exit(EXIT_FAILURE);
        }
        return *this;
    }

    /* Data accessing operators */
    auto operator[](const uint site) -> TField& { return _Field[site]; }
    auto operator[](const uint site) const -> const TField& { return _Field[site]; }

    /* Getters for lattice parameters */
    [[nodiscard]] auto getNt() const -> uint { return this->_Sizes[_t]; }
    [[nodiscard]] auto getNx() const -> uint { return this->_Sizes[_x]; }
    [[nodiscard]] auto getNy() const -> uint { return this->_Sizes[_y]; }
    [[nodiscard]] auto getNz() const -> uint { return this->_Sizes[_z]; }
    [[nodiscard]] auto getNi(uint Dir) const -> uint { return this->_Sizes[Dir]; }
    [[nodiscard]] auto getDim() const -> uint { return dim; }
    [[nodiscard]] auto getNsites() const -> uint { return this->_NSites; }
    [[nodiscard]] auto getVolume() const -> uint { return this->_NSites; }
    [[nodiscard]] auto getSubVols() const -> uintvect<dim> { return this->_SubVols; }
    [[nodiscard]] auto getSizes() const -> uintvect<dim> { return this->_Sizes; }

    /* Access next and previous sites */
    auto               next(uint site, uint dir) -> TField& { return _Field[this->_Next[site][dir]]; }
    auto               next(uint site, uint dir) const -> const TField& { return _Field[this->_Next[site][dir]]; }
    auto               nextId(uint site, uint dir) -> uint& { return this->_Next[site][dir]; }
    [[nodiscard]] auto nextId(uint site, uint dir) const -> const uint& { return this->_Next[site][dir]; }
    auto               prev(uint site, uint dir) -> TField& { return _Field[this->_Prev[site][dir]]; }
    auto               prev(uint site, uint dir) const -> const TField& { return _Field[this->_Prev[site][dir]]; }
    auto               prevId(uint site, uint dir) -> uint& { return this->_Prev[site][dir]; }
    [[nodiscard]] auto prevId(uint site, uint dir) const -> const uint& { return this->_Prev[site][dir]; }

    /* Get a memory report from Lattice */
    [[nodiscard]] auto memoryReport() const -> size_t {
        size_t Tot = 0;
        Tot += sizeof(TField) * _Field.size();  // byte size of the main lattice
        // Tot += sizeof(uint) * dim * Prev.size();  // byte size of the prev index lattice
        // Tot += sizeof(uint) * dim * Next.size();  // byte size of the next index lattice
        // Tot += sizeof(uint) * dim;                // byte size of the lattice sizes vector
        // Tot += sizeof(uint) * dim;                // byte size of the subvolume vector
        // Tot += sizeof(uint);                      // byte size of the number of lattice sizes

        return sizeof(TField) * _Field.size();
    }

    /* Save current field configuration */
    inline auto save_Configuration(const std::string& FilePath) -> hsize_t {
        hsize_t FileSize;
        try {
            // Open file
            H5::H5File File(FilePath, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            // Create dataspace
            std::array<hsize_t, 1> Dims = {this->NSites};  // dim sizes of ds (on disk)
            H5::DataSpace          DataSpace(1, Dims.data());
            // Create datatype
            auto DataType = make_H5_Type<TField>();
            // Create dataset
            H5::DataSet Dataset = File.createDataSet("field", DataType, DataSpace);
            // Write dataset
            Dataset.write(_Field.data(), DataType);
            // Get file size
            FileSize = File.getFileSize();
        } catch (H5::Exception& _) {
            exit(EXIT_FAILURE);
        }
        return FileSize;
    }

    /* Read configuration from disk */
    inline auto read_Configuration(const std::string& FilePath) -> hsize_t {
        hsize_t DataSize;
        try {
            // Open file
            H5::H5File File(FilePath, H5F_ACC_RDONLY, H5P_DEFAULT, H5P_DEFAULT);
            // Open DataSet
            H5::DataSet DataSet = File.openDataSet("field");
            // check DataTypes
            H5::DataType FileDataType = DataSet.getDataType();
            H5::DataType LatticeDataType = make_H5_Type<TField>();
            if (FileDataType != LatticeDataType) {
                H5::DataTypeIException Exeption("lattice::read_Configuration()", "File and data DataType differ");
                throw(Exeption);
            }
            // check DataSpaces
            H5::DataSpace FileDataSpace = DataSet.getSpace();
            hsize_t       FileDims = FileDataSpace.getSimpleExtentNdims();
            if (FileDims != this->NSites) {
                H5::DataSpaceIException Exeption("lattice::read_Configuration()", "File and data DataSpaces differ");
                throw(Exeption);
            }
            // read data
            DataSet.read(_Field.data(), LatticeDataType);
            // get data size
            DataSize = DataSet.getInMemDataSize();
        } catch (H5::DataTypeIException& e) {
            std::cerr << IO::LI_erro() << e.getFuncName() << " : " << e.getDetailMsg() << '\n';
            exit(EXIT_FAILURE);
        } catch (H5::DataSpaceIException& e) {
            std::cerr << IO::LI_erro() << e.getFuncName() << " : " << e.getDetailMsg() << '\n';
            exit(EXIT_FAILURE);
        }
        return DataSize;
    }
};

}  // namespace reticolo
