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
#include <string>
#include <vector>

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
    void saveLattice(const fs::path& FileName, const std::string& LatticeID, const Lattice<T>& field);
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
inline void HDF5Handler::saveLattice(const fs::path& FileName, const std::string& LatticeID, const Lattice<T>& field) {
    omp_set_lock(&_IoLock);
    std::array<hsize_t, 1> Entries = {field.size()};

    hid_t FileId = H5Fcreate(FileName.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
    /* write field as dataset */
    hid_t DataSpaceId = H5Screate_simple(1, Entries.data(), H5P_DEFAULT);
    hid_t DataTypeId = make_H5_Type<T>();
    hid_t DataSetId =
        H5Dcreate(FileId, LatticeID.c_str(), DataTypeId, DataSpaceId, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(DataSetId, DataTypeId, H5S_ALL, H5S_ALL, H5P_DEFAULT, field.data());
    /* write lattice sizes as attribute */
    std::array<hsize_t, 1> Dimension = {(hsize_t)field.getDim()};
    hid_t                  AttrSpaceId = H5Screate_simple(1, Dimension.data(), H5P_DEFAULT);
    hid_t                  AttrTypeId = make_H5_Type<typename Lattice<T>::size_type>();
    hid_t                  AttrId = H5Acreate(DataSetId, "sizes", AttrTypeId, AttrSpaceId, H5P_DEFAULT, H5P_DEFAULT);
    H5Awrite(AttrId, AttrTypeId, field.getSizes().data());

    H5close();
    omp_unset_lock(&_IoLock);
}

inline HDF5Handler GlobalHdf5Handler;
}  // namespace reticolo