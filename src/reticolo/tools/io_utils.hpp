/******************************************************************************

 - reticolo (www.github.com/olmo-francesconi/reticolo.git)

 - SourceFile: tools/io_utils.hpp

 - Author: Olmo Francesconi <olmo.francesconi@glasgow.ac.uk>

******************************************************************************/

#pragma once

#include <H5Dpublic.h>
#include <H5Fpublic.h>
#include <H5Gpublic.h>
#include <H5Ipublic.h>
#include <H5Ppublic.h>
#include <H5Spublic.h>
#include <H5public.h>
#include <H5version.h>
#include <hdf5.h>
#include <omp.h>

#include <array>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <format>
#include <sstream>
#include <string>
#include <vector>

#include "reticolo/tools/timer.hpp"
#include "reticolo/types/concepts.hpp"  // IWYU pragma: keep
#include "reticolo/types/core.hpp"

namespace reticolo::IO {

/*--------------------------------------------------------------------------------------------------
    Styling
--------------------------------------------------------------------------------------------------*/

/* Returns the reticolo Welcome screen as a std::string */
inline auto pretty_welcome() -> std::string {
    // clang-format off
    const std::string WelcomeLogo =
        R"(________________________________________________________________________________)" "\n"
        R"(                                                                                )" "\n"
        R"(         ██████╗ ███████╗████████╗██╗ ██████╗ ██████╗ ██╗      ██████╗          )" "\n"
        R"(         ██╔══██╗██╔════╝╚══██╔══╝██║██╔════╝██╔═══██╗██║     ██╔═══██╗         )" "\n"
        R"(         ██████╔╝█████╗     ██║   ██║██║     ██║   ██║██║     ██║   ██║         )" "\n"
        R"(         ██╔══██╗██╔══╝     ██║   ██║██║     ██║   ██║██║     ██║   ██║         )" "\n"
        R"(         ██║  ██║███████╗   ██║   ██║╚██████╗╚██████╔╝███████╗╚██████╔╝         )" "\n"
        R"(         ╚═╝  ╚═╝╚══════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝ ╚═════╝          )" "\n"
        R"(________________________________________________________________________________)" "\n";
    // clang-format on
    return WelcomeLogo;
}

const size_t KILO = 1024;  // Definition of what a Kilo of stuff is (Decimal: 1000, Binary: 1024)
/* Returns a std::string representing the byte size in a convenient unit*/
inline auto pretty_bytes(size_t bytes) -> std::string {
    std::array<std::string, 7> Suffixes({" B", "KB", "MB", "GB", "TB", "PB", "EB"});

    uint   SuffixIndex = 0;  // Pretty suffix index
    double Count = bytes;    // Pretty value
    while (Count >= KILO && SuffixIndex < Suffixes.size()) {
        SuffixIndex++;
        Count /= KILO;
    }
    return std::format("{:>7.2f}", Count) + " " + Suffixes[SuffixIndex];
}

/*--------------------------------------------------------------------------------------------------
    Logging helper functions
--------------------------------------------------------------------------------------------------*/

/* Default reticolo log line init with timing */
inline auto LI_time() -> std::string {
    auto        Time = std::chrono::duration<double>(GlobalTimer.elapsed_s());
    std::string Message = "reticolo......." + std::format("{:%T}", Time) + " | ";
    return Message;
}

/* Default reticolo log line init with dots */
inline auto LI_dots() -> std::string {
    std::string Message = "reticolo............... | ";
    return Message;
}

/* Default reticolo log line empty init */
inline auto LI_void() -> std::string {
    std::string Message = "                        | ";
    return Message;
}

/* Default reticolo log error line init */
inline auto LI_erro() -> std::string {
    std::string Message = "reticolo..........ERROR | ";
    return Message;
}

/* Default reticolo log warning line init */
inline auto LI_warn() -> std::string {
    std::string Message = "reticolo........WARNING | ";
    return Message;
}

/*--------------------------------------------------------------------------------------------------
    Generic print() functions for the various types
--------------------------------------------------------------------------------------------------*/

/* Print Reals in standard format (signed 8 digits scientific) */
template <RealValue T>
auto print(T val) -> std::string {
    return std::format("{:+8e}", val);
}

/* Print Complex numbers in standard format (dual signed 8 digits scientific) */
template <ComplexValue T>
auto print(T val) -> std::string {
    return std::format("{:+8e}{:+8e}I", val.real(), val.imag());
}

/* Print uint Vectors in standard format */
template <size_t dim>
inline auto print(const intvect<dim>& Vect) -> std::string {
    std::stringstream Res;
    Res << "[" << Vect[0];
    for (uint Comp = 1; Comp < dim; Comp++) {
        Res << " x " << Vect[Comp];
    }
    Res << "]";
    return Res.str();
}

/*--------------------------------------------------------------------------------------------------
    Hdf5Handler class
--------------------------------------------------------------------------------------------------*/

namespace fs = std::filesystem;

class HdF5Handler {
  private:
    omp_lock_t _IoLock;

  public:
    HdF5Handler() { omp_init_lock(&_IoLock); };
    ~HdF5Handler() { omp_destroy_lock(&_IoLock); }

    /* File creation */
    void initFile(const fs::path& FileName);
    auto checkFile(const fs::path& FileName) -> bool;

    /* Create group */
    void createGroup(const fs::path& FileName, const std::string& GrouName);

    /* Write to simple dataset */
    template <typename T>
    void writeDataset(const fs::path& FileName, const std::string& DataSetName, const std::vector<T>& data);

    /* Write to extendable dataset */
    template <typename T>
    void setupExpandableDataset(const fs::path& FileName, const std::string& DataSetName, hsize_t ChunkSize);
    template <typename T>
    void appendDataset(const fs::path& FileName, const std::string& DataSetName, const std::vector<T>& data);

    // template <typename T>
    // void appendDataset(const fs::path& FileName, const std::string& DataSetName, const std::vector<T>& data);
};

void HdF5Handler::initFile(const fs::path& FileName) {
    omp_set_lock(&_IoLock);
    hid_t FileId = H5Fcreate(FileName.c_str(), H5F_ACC_EXCL, H5P_DEFAULT, H5P_DEFAULT);
    H5Fclose(FileId);
    omp_unset_lock(&_IoLock);
}

auto HdF5Handler::checkFile(const fs::path& FileName) -> bool {
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

void HdF5Handler::createGroup(const fs::path& FileName, const std::string& GrouName) {
    omp_set_lock(&_IoLock);
    hid_t FileId = H5Fopen(FileName.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
    hid_t GroupId = H5Gcreate(FileId, GrouName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Gclose(GroupId);
    H5Fclose(FileId);
    omp_unset_lock(&_IoLock);
}

template <typename T>
void HdF5Handler::writeDataset(const fs::path& FileName, const std::string& DataSetName, const std::vector<T>& data) {
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
void HdF5Handler::setupExpandableDataset(const fs::path& FileName, const std::string& DataSetName, hsize_t ChunkSize) {
    omp_set_lock(&_IoLock);
    hid_t FileId = H5Fopen(FileName.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

    hid_t DataSpaceId =
        H5Screate_simple(1, (std::array<hsize_t, 1>){0}.data(), (std::array<hsize_t, 1>){H5S_UNLIMITED}.data());
    hid_t DsCreationPropId = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(DsCreationPropId, 1, (std::array<hsize_t, 1>){ChunkSize}.data());
    hid_t DataTypeId = make_H5_Type<T>();
    hid_t DataSetId =
        H5Dcreate(FileId, DataSetName.c_str(), DataTypeId, DataSpaceId, H5P_DEFAULT, DsCreationPropId, H5P_DEFAULT);
    H5close();
    omp_unset_lock(&_IoLock);
}

template <typename T>
void HdF5Handler::appendDataset(const fs::path& FileName, const std::string& DataSetName, const std::vector<T>& data) {
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
    H5Sselect_hyperslab(DataSpaceID, H5S_SELECT_SET, OldSize.data(), NULL, ExtendSize.data(), NULL);
    hid_t BufferDataSpaceID = H5Screate_simple(1, ExtendSize.data(), ExtendSize.data());
    H5Dwrite(DataSetId, DataTypeId, BufferDataSpaceID, DataSpaceID, H5P_DEFAULT, data.data());
    H5close();
    omp_unset_lock(&_IoLock);
}

inline HdF5Handler GlobalHdf5Handler;

}  // namespace reticolo::IO
