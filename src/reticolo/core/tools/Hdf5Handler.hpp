#pragma once

#include <H5Apublic.h>
#include <H5Dpublic.h>
#include <H5Fpublic.h>
#include <H5Gpublic.h>
#include <H5Ipublic.h>
#include <H5Ppublic.h>
#include <H5Spublic.h>
#include <H5Tpublic.h>
#include <H5public.h>
#include <H5version.h>
#include <omp.h>

#include <array>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "reticolo/lattice/indexing.hpp"
#include "reticolo/lattice/lattice.hpp"

namespace reticolo {
/*--------------------------------------------------------------------------------------------------
    Hdf5Handler class
--------------------------------------------------------------------------------------------------*/

namespace fs = std::filesystem;

class HDF5Handler {
  private:
    omp_lock_t _IoLock;

  public:
    HDF5Handler() { omp_init_lock(&_IoLock); };
    ~HDF5Handler() { omp_destroy_lock(&_IoLock); }

    /* File creation */
    void initFile(const fs::path& FileName);
    auto checkFile(const fs::path& FileName) -> bool;

    /* Create group */
    void createGroup(const fs::path& FileName, const std::string& GroupName);

    /* Write to simple dataset */
    template <typename T>
    void writeDataset(const fs::path& FileName, const std::string& DataSetName, const std::vector<T>& data);

    /* Write to extendable dataset */
    template <typename T>
    void setupExpandableDataset(const fs::path& FileName, const std::string& DataSetName, hsize_t ChunkSize,
                                bool Compressed);
    template <typename T>
    void appendDataset(const fs::path& FileName, const std::string& DataSetName, const std::vector<T>& data);

    template <typename T>
    void saveLattice(const fs::path& FileName, const std::string& LatticeID, const Lattice<T>& field,
                     const std::stringstream& rngState);

    template <typename T>
    void readLattice(const fs::path& FileName, const std::string& LatticeID, Lattice<T>& field,
                     std::stringstream& RngState);
};

inline void HDF5Handler::initFile(const fs::path& FileName) {
    omp_set_lock(&_IoLock);
    hid_t FileId = H5Fcreate(FileName.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
    H5Fclose(FileId);
    omp_unset_lock(&_IoLock);
}

inline auto HDF5Handler::checkFile(const fs::path& FileName) -> bool {
    omp_set_lock(&_IoLock);
    hid_t FileId = H5Fopen(FileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (FileId == H5I_INVALID_HID) {
        omp_unset_lock(&_IoLock);
        return false;
    }
    H5Fclose(FileId);
    omp_unset_lock(&_IoLock);
    return true;
}

inline void HDF5Handler::createGroup(const fs::path& FileName, const std::string& GroupName) {
    omp_set_lock(&_IoLock);
    hid_t FileId = H5Fopen(FileName.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    hid_t GroupId = H5Gcreate(FileId, GroupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Gclose(GroupId);
    H5Fclose(FileId);
    omp_unset_lock(&_IoLock);
}

template <typename T>
inline void HDF5Handler::writeDataset(const fs::path& FileName, const std::string& DataSetName,
                                      const std::vector<T>& data) {
    omp_set_lock(&_IoLock);
    std::array<hsize_t, 1> Entries = {data.size()};

    hid_t FileId = H5Fopen(FileName.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    hid_t DataSpaceId = H5Screate_simple(1, Entries.data(), H5P_DEFAULT);
    hid_t DataTypeId = make_H5_Type<T>();
    hid_t DataSetId =
        H5Dcreate(FileId, DataSetName.c_str(), DataTypeId, DataSpaceId, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(DataSetId, DataTypeId, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
    H5close();
    omp_unset_lock(&_IoLock);
}

template <typename T>
inline void HDF5Handler::setupExpandableDataset(const fs::path& FileName, const std::string& DataSetName,
                                                hsize_t ChunkSize, bool Compressed) {
    omp_set_lock(&_IoLock);
    hid_t FileId = H5Fopen(FileName.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

    hid_t DataSpaceId =
        H5Screate_simple(1, (std::array<hsize_t, 1>){0}.data(), (std::array<hsize_t, 1>){H5S_UNLIMITED}.data());
    hid_t DsCreationPropId = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(DsCreationPropId, 1, (std::array<hsize_t, 1>){ChunkSize}.data());
    if (Compressed) {
        H5Pset_deflate(DsCreationPropId, 2);
    }
    hid_t DataTypeId = make_H5_Type<T>();
    H5Dcreate(FileId, DataSetName.c_str(), DataTypeId, DataSpaceId, H5P_DEFAULT, DsCreationPropId, H5P_DEFAULT);
    H5close();
    omp_unset_lock(&_IoLock);
}

template <typename T>
inline void HDF5Handler::appendDataset(const fs::path& FileName, const std::string& DataSetName,
                                       const std::vector<T>& data) {
    omp_set_lock(&_IoLock);
    std::array<hsize_t, 1> OldSize;
    std::array<hsize_t, 1> NewSize;
    std::array<hsize_t, 1> ExtendSize = {data.size()};
    std::array<hsize_t, 1> MaxSize;

    hid_t FileId = H5Fopen(FileName.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    hid_t DataSetId = H5Dopen(FileId, DataSetName.c_str(), H5P_DEFAULT);
    hid_t DataTypeId = make_H5_Type<T>();
    hid_t DataSpaceID = H5Dget_space(DataSetId);
    H5Sget_simple_extent_dims(DataSpaceID, OldSize.data(), MaxSize.data());
    NewSize[0] = OldSize[0] + ExtendSize[0];
    H5Dset_extent(DataSetId, NewSize.data());
    DataSpaceID = H5Dget_space(DataSetId);
    H5Sselect_hyperslab(DataSpaceID, H5S_SELECT_SET, OldSize.data(), nullptr, ExtendSize.data(), nullptr);
    hid_t BufferDataSpaceID = H5Screate_simple(1, ExtendSize.data(), ExtendSize.data());
    H5Dwrite(DataSetId, DataTypeId, BufferDataSpaceID, DataSpaceID, H5P_DEFAULT, data.data());
    H5close();
    omp_unset_lock(&_IoLock);
}

template <typename T>
inline void HDF5Handler::saveLattice(const fs::path& FileName, const std::string& LatticeID, const Lattice<T>& field,
                                     const std::stringstream& RngState) {
    /* Get a omp_lock */
    omp_set_lock(&_IoLock);
    /* create HDF5 file*/
    hid_t FileId = H5Fcreate(FileName.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
    /* write field as dataset */
    std::array<hsize_t, 1> Entries = {field.size()};
    hid_t                  DataSpaceId = H5Screate_simple(1, Entries.data(), H5P_DEFAULT);
    hid_t                  DataTypeId = make_H5_Type<T>();
    hid_t                  DataSetId =
        H5Dcreate2(FileId, LatticeID.c_str(), DataTypeId, DataSpaceId, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(DataSetId, DataTypeId, H5S_ALL, H5S_ALL, H5P_DEFAULT, field.data());
    /* write lattice sizes as attribute */
    std::array<hsize_t, 1> SizeAttrDim = {(hsize_t)field.getDim()};
    hid_t                  SizeAttrSpaceId = H5Screate_simple(1, SizeAttrDim.data(), H5P_DEFAULT);
    hid_t                  SizeAttrTypeId = make_H5_Type<typename Lattice<T>::size_type>();
    hid_t SizeAttrId = H5Acreate(DataSetId, "sizes", SizeAttrTypeId, SizeAttrSpaceId, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(SizeAttrId, SizeAttrTypeId, field.getSizes().data());
    /* write RngState as attribute */
    std::string State = RngState.str();
    hid_t       RngAttrSpaceId = H5Screate(H5S_SCALAR);
    hid_t       RngAttrTypeId = H5Tcopy(H5T_C_S1);
    H5Tset_size(RngAttrTypeId, State.size() + 1);
    H5Tset_strpad(RngAttrTypeId, H5T_STR_NULLTERM);
    hid_t RngAttrId = H5Acreate(DataSetId, "RngState", RngAttrTypeId, RngAttrSpaceId, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(RngAttrId, RngAttrTypeId, State.c_str());
    /* Deallocate all HDF5 resources*/
    H5close();
    /* Release the omp_lock */
    omp_unset_lock(&_IoLock);
}

template <typename T>
inline void HDF5Handler::readLattice(const fs::path& FileName, const std::string& LatticeID, Lattice<T>& field,
                                     std::stringstream& RngState) {
    /* Get a omp_lock */
    omp_set_lock(&_IoLock);
    /* Open HDF5 file*/
    hid_t FileId = H5Fopen(FileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    /* Read field dataset */
    // open the dataset
    hid_t DataSetId = H5Dopen(FileId, LatticeID.c_str(), H5P_DEFAULT);
    // check the dataspace is one-dimensional
    hid_t     DataSpaceId = H5Dget_space(DataSetId);
    const int DataSpaceNDims = H5Sget_simple_extent_ndims(DataSpaceId);
    if (DataSpaceNDims != 1) {
        throw std::runtime_error("HDF5Handler - Error while reading configuration (dataspace mismatch)");
    }
    // check the size of the dataset
    std::vector<hsize_t> Entries(DataSpaceNDims);
    H5Sget_simple_extent_dims(DataSpaceId, Entries.data(), nullptr);
    if (Entries[0] != field.size()) {
        throw std::runtime_error("HDF5Handler - Error while reading configuration (dimension mismatch)");
    }
    // check the sizes attribute
    hid_t                SizeAttrId = H5Aopen(DataSetId, "sizes", H5P_DEFAULT);
    hid_t                SizeAttrSpaceId = H5Aget_space(SizeAttrId);
    const int            SizeAttrNDims = H5Sget_simple_extent_ndims(SizeAttrSpaceId);
    std::vector<hsize_t> SizeAttrDims(SizeAttrNDims, 0);
    H5Sget_simple_extent_dims(SizeAttrSpaceId, SizeAttrDims.data(), nullptr);
    if (SizeAttrDims[0] != field.getDim()) {
        throw std::runtime_error("HDF5Handler - Error while reading configuration (dimension mismatch)");
    }
    std::vector<Indexing::size_type> Sizes(SizeAttrDims[0], 0);
    hid_t                            SizeAttrTypeId = H5Aget_type(SizeAttrId);
    hid_t                            SizeAttrTypeIdCheck = make_H5_Type<typename Lattice<T>::size_type>();
    if (H5Tequal(SizeAttrTypeId, SizeAttrTypeIdCheck) != 1) {
        throw std::runtime_error("HDF5Handler - Error while reading configuration (sizes attribute type mismatch)");
    }
    H5Aread(SizeAttrId, SizeAttrTypeId, Sizes.data());
    if (Sizes != field.getSizes()) {
        throw std::runtime_error("HDF5Handler - Error while reading configuration (dimension mismatch)");
    }
    // check the field types
    hid_t DataTypeId = H5Dget_type(DataSetId);
    hid_t DataTypeIdCheck = make_H5_Type<T>();
    if (H5Tequal(DataTypeId, DataTypeIdCheck) != 1) {
        throw std::runtime_error("HDF5Handler - Error while reading configuration (field type mismatch)");
    }
    // read the dataset
    H5Dread(DataSetId, DataTypeId, DataSpaceId, DataSpaceId, H5P_DEFAULT, field.data());

    /* Read RngState as attribute */
    hid_t             RngAttrId = H5Aopen(DataSetId, "RngState", H5P_DEFAULT);
    hsize_t           RngAttrSize = H5Aget_storage_size(RngAttrId);
    std::vector<char> RngRaw(RngAttrSize + 1);
    hid_t             RngAttrType = H5Aget_type(RngAttrId);
    H5Aread(RngAttrId, RngAttrType, RngRaw.data());
    RngState.str(std::string());
    RngState << std::string(RngRaw.begin(), RngRaw.end());

    /* Deallocate all HDF5 resources*/
    H5close();
    /* Release the omp_lock */
    omp_unset_lock(&_IoLock);
}

inline HDF5Handler GlobalHdf5Handler;
}  // namespace reticolo