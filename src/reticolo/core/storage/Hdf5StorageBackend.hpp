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

#include <array>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "reticolo/core/storage/Hdf5TypeMappings.hpp"
#include "reticolo/core/storage/StorageSchema.hpp"
#include "reticolo/core/tools/omp_compat.hpp"
#include "reticolo/lattice/indexing.hpp"
#include "reticolo/lattice/lattice.hpp"

namespace reticolo::storage {

namespace fs = std::filesystem;

namespace detail {

class OmpLockGuard {
  public:
    explicit OmpLockGuard(omp_lock_t* lock) : _Lock(lock) { omp_set_lock(_Lock); }
    ~OmpLockGuard() {
        if (_Lock != nullptr) {
            omp_unset_lock(_Lock);
        }
    }
    OmpLockGuard(const OmpLockGuard&) = delete;
    auto operator=(const OmpLockGuard&) -> OmpLockGuard& = delete;

  private:
    omp_lock_t* _Lock;
};

template <herr_t (*Closer)(hid_t)>
class H5Scoped {
  public:
    H5Scoped() = default;
    explicit H5Scoped(hid_t hid) : _Id(hid) {}
    ~H5Scoped() { reset(); }

    H5Scoped(const H5Scoped&) = delete;
    auto operator=(const H5Scoped&) -> H5Scoped& = delete;

    H5Scoped(H5Scoped&& other) noexcept : _Id(std::exchange(other._Id, H5I_INVALID_HID)) {}
    auto operator=(H5Scoped&& other) noexcept -> H5Scoped& {
        if (this != &other) {
            reset();
            _Id = std::exchange(other._Id, H5I_INVALID_HID);
        }
        return *this;
    }

    [[nodiscard]] auto     get() const -> hid_t { return _Id; }
    [[nodiscard]] explicit operator bool() const { return _Id >= 0; }

    void reset(hid_t new_id = H5I_INVALID_HID) {
        if (_Id >= 0) {
            (void)Closer(_Id);
        }
        _Id = new_id;
    }

  private:
    hid_t _Id = H5I_INVALID_HID;
};

using H5File = H5Scoped<&H5Fclose>;
using H5Group = H5Scoped<&H5Gclose>;
using H5Dataset = H5Scoped<&H5Dclose>;
using H5Dataspace = H5Scoped<&H5Sclose>;
using H5Attr = H5Scoped<&H5Aclose>;
using H5Prop = H5Scoped<&H5Pclose>;

class H5Type {
  public:
    H5Type() = default;
    H5Type(hid_t hid, bool owned) : _Id(hid), _Owned(owned) {}
    ~H5Type() { reset(); }

    H5Type(const H5Type&) = delete;
    auto operator=(const H5Type&) -> H5Type& = delete;

    H5Type(H5Type&& other) noexcept : _Id(std::exchange(other._Id, H5I_INVALID_HID)), _Owned(other._Owned) {
        other._Owned = false;
    }
    auto operator=(H5Type&& other) noexcept -> H5Type& {
        if (this != &other) {
            reset();
            _Id = std::exchange(other._Id, H5I_INVALID_HID);
            _Owned = other._Owned;
            other._Owned = false;
        }
        return *this;
    }

    [[nodiscard]] auto     get() const -> hid_t { return _Id; }
    [[nodiscard]] explicit operator bool() const { return _Id >= 0; }

    void reset(hid_t new_id = H5I_INVALID_HID, bool owned = false) {
        if (_Owned && _Id >= 0) {
            (void)H5Tclose(_Id);
        }
        _Id = new_id;
        _Owned = owned;
    }

  private:
    hid_t _Id = H5I_INVALID_HID;
    bool  _Owned = false;
};

template <typename T>
struct h5_type_is_predefined : std::false_type {};

template <>
struct h5_type_is_predefined<unsigned short> : std::true_type {};
template <>
struct h5_type_is_predefined<unsigned int> : std::true_type {};
template <>
struct h5_type_is_predefined<size_t> : std::true_type {};
template <>
struct h5_type_is_predefined<float> : std::true_type {};
template <>
struct h5_type_is_predefined<double> : std::true_type {};

inline void ensure_valid(hid_t hid, const char* what) {
    if (hid < 0) {
        throw std::runtime_error(std::string("Hdf5StorageBackend - HDF5 call failed: ") + what);
    }
}

inline auto link_exists(hid_t loc_id, const std::string& path) -> bool {
    const htri_t res = H5Lexists(loc_id, path.c_str(), H5P_DEFAULT);
    if (res < 0) {
        throw std::runtime_error(std::string("Hdf5StorageBackend - HDF5 call failed: H5Lexists(") + path + ")");
    }
    return res > 0;
}

inline auto parent_path(std::string_view path) -> std::string {
    const auto pos = path.find_last_of('/');
    if (pos == std::string_view::npos) {
        return {};
    }
    return std::string(path.substr(0, pos));
}

inline void ensure_groups(hid_t file_id, std::string_view object_path) {
    const auto group_path = parent_path(object_path);
    if (group_path.empty()) {
        return;
    }

    std::size_t start = 0;
    while (start < group_path.size()) {
        const auto end = group_path.find('/', start);
        const auto len = (end == std::string::npos) ? group_path.size() - start : end - start;
        if (len > 0) {
            const auto current = group_path.substr(0, start + len);
            if (!link_exists(file_id, current)) {
                H5Group group_id(H5Gcreate2(file_id, current.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
                ensure_valid(group_id.get(), "H5Gcreate2(parent group)");
            }
        }
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
}

template <typename T>
inline auto make_type() -> H5Type {
    hid_t hid = make_H5_Type<T>();
    ensure_valid(hid, "make_H5_Type<T>()");
    return H5Type{hid, !h5_type_is_predefined<T>::value};
}

inline auto make_file_access_props_for_create() -> H5Prop {
    H5Prop props(H5Pcreate(H5P_FILE_ACCESS));
    ensure_valid(props.get(), "H5Pcreate(H5P_FILE_ACCESS)");
#if defined(RETICOLO_HDF5_NATIVE_COMPLEX) && RETICOLO_HDF5_NATIVE_COMPLEX
    if constexpr (hdf5::supports_native_complex()) {
        ensure_valid(H5Pset_libver_bounds(props.get(), H5F_LIBVER_V200, H5F_LIBVER_LATEST),
                     "H5Pset_libver_bounds(V200)");
    }
#endif
    return props;
}

}  // namespace detail

class Hdf5StorageBackend {
  private:
    omp_lock_t _IoLock;

  public:
    Hdf5StorageBackend() { omp_init_lock(&_IoLock); };
    ~Hdf5StorageBackend() { omp_destroy_lock(&_IoLock); }

    void initFile(const fs::path& FileName);
    auto checkFile(const fs::path& FileName) -> bool;
    void createGroup(const fs::path& FileName, const std::string& GroupName);

    template <typename T>
    void writeDataset(const fs::path& FileName, const std::string& DataSetName, const std::vector<T>& data);

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

inline void Hdf5StorageBackend::initFile(const fs::path& FileName) {
    detail::OmpLockGuard Lock(&_IoLock);
    auto                 FileAccessProps = detail::make_file_access_props_for_create();
    hid_t                RawFileId = H5Fcreate(FileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, FileAccessProps.get());
    detail::ensure_valid(RawFileId, "H5Fcreate");
    detail::H5File FileId(RawFileId);
}

inline auto Hdf5StorageBackend::checkFile(const fs::path& FileName) -> bool {
    detail::OmpLockGuard Lock(&_IoLock);
    hid_t                RawFileId = H5Fopen(FileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (RawFileId < 0 || RawFileId == H5I_INVALID_HID) {
        return false;
    }
    detail::H5File FileId(RawFileId);
    return true;
}

inline void Hdf5StorageBackend::createGroup(const fs::path& FileName, const std::string& GroupName) {
    detail::OmpLockGuard Lock(&_IoLock);
    detail::H5File       FileId(H5Fopen(FileName.c_str(), H5F_ACC_RDWR, H5P_DEFAULT));
    detail::ensure_valid(FileId.get(), "H5Fopen");
    if (detail::link_exists(FileId.get(), GroupName)) {
        return;
    }
    detail::H5Group GroupId(H5Gcreate2(FileId.get(), GroupName.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    detail::ensure_valid(GroupId.get(), "H5Gcreate2");
}

template <typename T>
inline void Hdf5StorageBackend::writeDataset(const fs::path& FileName, const std::string& DataSetName,
                                             const std::vector<T>& data) {
    detail::OmpLockGuard   Lock(&_IoLock);
    std::array<hsize_t, 1> Entries = {static_cast<hsize_t>(data.size())};
    detail::H5File         FileId(H5Fopen(FileName.c_str(), H5F_ACC_RDWR, H5P_DEFAULT));
    detail::ensure_valid(FileId.get(), "H5Fopen");
    if (detail::link_exists(FileId.get(), DataSetName)) {
        detail::ensure_valid(H5Ldelete(FileId.get(), DataSetName.c_str(), H5P_DEFAULT), "H5Ldelete(dataset)");
    }
    detail::H5Dataspace DataSpaceId(H5Screate_simple(1, Entries.data(), nullptr));
    detail::ensure_valid(DataSpaceId.get(), "H5Screate_simple");
    auto              DataTypeId = detail::make_type<T>();
    detail::H5Dataset DataSetId(H5Dcreate2(FileId.get(), DataSetName.c_str(), DataTypeId.get(), DataSpaceId.get(),
                                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    detail::ensure_valid(DataSetId.get(), "H5Dcreate2");
    detail::ensure_valid(H5Dwrite(DataSetId.get(), DataTypeId.get(), H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data()),
                         "H5Dwrite");
}

template <typename T>
inline void Hdf5StorageBackend::setupExpandableDataset(const fs::path& FileName, const std::string& DataSetName,
                                                       hsize_t ChunkSize, bool Compressed) {
    detail::OmpLockGuard Lock(&_IoLock);
    detail::H5File       FileId(H5Fopen(FileName.c_str(), H5F_ACC_RDWR, H5P_DEFAULT));
    detail::ensure_valid(FileId.get(), "H5Fopen");
    if (detail::link_exists(FileId.get(), DataSetName)) {
        detail::ensure_valid(H5Ldelete(FileId.get(), DataSetName.c_str(), H5P_DEFAULT), "H5Ldelete(expandable ds)");
    }
    std::array<hsize_t, 1> CurrentDims = {0};
    std::array<hsize_t, 1> MaxDims = {H5S_UNLIMITED};
    detail::H5Dataspace    DataSpaceId(H5Screate_simple(1, CurrentDims.data(), MaxDims.data()));
    detail::ensure_valid(DataSpaceId.get(), "H5Screate_simple");
    detail::H5Prop DsCreationPropId(H5Pcreate(H5P_DATASET_CREATE));
    detail::ensure_valid(DsCreationPropId.get(), "H5Pcreate(H5P_DATASET_CREATE)");
    std::array<hsize_t, 1> ChunkDims = {ChunkSize};
    detail::ensure_valid(H5Pset_chunk(DsCreationPropId.get(), 1, ChunkDims.data()), "H5Pset_chunk");
    if (Compressed) {
        detail::ensure_valid(H5Pset_deflate(DsCreationPropId.get(), 2), "H5Pset_deflate");
    }
    auto              DataTypeId = detail::make_type<T>();
    detail::H5Dataset DataSetId(H5Dcreate2(FileId.get(), DataSetName.c_str(), DataTypeId.get(), DataSpaceId.get(),
                                           H5P_DEFAULT, DsCreationPropId.get(), H5P_DEFAULT));
    detail::ensure_valid(DataSetId.get(), "H5Dcreate2");
}

template <typename T>
inline void Hdf5StorageBackend::appendDataset(const fs::path& FileName, const std::string& DataSetName,
                                              const std::vector<T>& data) {
    detail::OmpLockGuard   Lock(&_IoLock);
    std::array<hsize_t, 1> OldSize = {0};
    std::array<hsize_t, 1> NewSize = {0};
    std::array<hsize_t, 1> ExtendSize = {static_cast<hsize_t>(data.size())};
    std::array<hsize_t, 1> MaxSize = {0};
    detail::H5File         FileId(H5Fopen(FileName.c_str(), H5F_ACC_RDWR, H5P_DEFAULT));
    detail::ensure_valid(FileId.get(), "H5Fopen");
    detail::H5Dataset DataSetId(H5Dopen2(FileId.get(), DataSetName.c_str(), H5P_DEFAULT));
    detail::ensure_valid(DataSetId.get(), "H5Dopen2");
    auto                DataTypeId = detail::make_type<T>();
    detail::H5Dataspace DataSpaceID(H5Dget_space(DataSetId.get()));
    detail::ensure_valid(DataSpaceID.get(), "H5Dget_space");
    detail::ensure_valid(H5Sget_simple_extent_dims(DataSpaceID.get(), OldSize.data(), MaxSize.data()),
                         "H5Sget_simple_extent_dims");
    NewSize[0] = OldSize[0] + ExtendSize[0];
    detail::ensure_valid(H5Dset_extent(DataSetId.get(), NewSize.data()), "H5Dset_extent");
    DataSpaceID.reset(H5Dget_space(DataSetId.get()));
    detail::ensure_valid(DataSpaceID.get(), "H5Dget_space(after set_extent)");
    detail::ensure_valid(
        H5Sselect_hyperslab(DataSpaceID.get(), H5S_SELECT_SET, OldSize.data(), nullptr, ExtendSize.data(), nullptr),
        "H5Sselect_hyperslab");
    detail::H5Dataspace BufferDataSpaceID(H5Screate_simple(1, ExtendSize.data(), ExtendSize.data()));
    detail::ensure_valid(BufferDataSpaceID.get(), "H5Screate_simple(buffer)");
    detail::ensure_valid(H5Dwrite(DataSetId.get(), DataTypeId.get(), BufferDataSpaceID.get(), DataSpaceID.get(),
                                  H5P_DEFAULT, data.data()),
                         "H5Dwrite");
}

template <typename T>
inline void Hdf5StorageBackend::saveLattice(const fs::path& FileName, const std::string& LatticeID,
                                            const Lattice<T>& field, const std::stringstream& RngState) {
    detail::OmpLockGuard Lock(&_IoLock);
    detail::H5File       FileId;
    if (fs::exists(FileName)) {
        FileId.reset(H5Fopen(FileName.c_str(), H5F_ACC_RDWR, H5P_DEFAULT));
        detail::ensure_valid(FileId.get(), "H5Fopen");
    } else {
        auto FileAccessProps = detail::make_file_access_props_for_create();
        FileId.reset(H5Fcreate(FileName.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, FileAccessProps.get()));
        detail::ensure_valid(FileId.get(), "H5Fcreate");
    }
    detail::ensure_groups(FileId.get(), LatticeID);
    if (detail::link_exists(FileId.get(), LatticeID)) {
        detail::ensure_valid(H5Ldelete(FileId.get(), LatticeID.c_str(), H5P_DEFAULT), "H5Ldelete(lattice)");
    }
    std::array<hsize_t, 1> Entries = {static_cast<hsize_t>(field.size())};
    detail::H5Dataspace    DataSpaceId(H5Screate_simple(1, Entries.data(), nullptr));
    detail::ensure_valid(DataSpaceId.get(), "H5Screate_simple");
    auto              DataTypeId = detail::make_type<T>();
    detail::H5Dataset DataSetId(H5Dcreate2(FileId.get(), LatticeID.c_str(), DataTypeId.get(), DataSpaceId.get(),
                                           H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    detail::ensure_valid(DataSetId.get(), "H5Dcreate2");
    detail::ensure_valid(H5Dwrite(DataSetId.get(), DataTypeId.get(), H5S_ALL, H5S_ALL, H5P_DEFAULT, field.data()),
                         "H5Dwrite");
    std::array<hsize_t, 1> SizeAttrDim = {static_cast<hsize_t>(field.getDim())};
    detail::H5Dataspace    SizeAttrSpaceId(H5Screate_simple(1, SizeAttrDim.data(), nullptr));
    detail::ensure_valid(SizeAttrSpaceId.get(), "H5Screate_simple(sizes attr)");
    auto           SizeAttrTypeId = detail::make_type<typename Lattice<T>::size_type>();
    detail::H5Attr SizeAttrId(H5Acreate2(DataSetId.get(), schema::lattice::field_sizes_attribute.data(),
                                         SizeAttrTypeId.get(), SizeAttrSpaceId.get(), H5P_DEFAULT, H5P_DEFAULT));
    detail::ensure_valid(SizeAttrId.get(), "H5Acreate2(sizes)");
    detail::ensure_valid(H5Awrite(SizeAttrId.get(), SizeAttrTypeId.get(), field.getSizes().data()), "H5Awrite(sizes)");
    std::string         State = RngState.str();
    detail::H5Dataspace RngAttrSpaceId(H5Screate(H5S_SCALAR));
    detail::ensure_valid(RngAttrSpaceId.get(), "H5Screate(H5S_SCALAR)");
    detail::H5Type RngAttrTypeId(H5Tcopy(H5T_C_S1), true);
    detail::ensure_valid(RngAttrTypeId.get(), "H5Tcopy(H5T_C_S1)");
    detail::ensure_valid(H5Tset_size(RngAttrTypeId.get(), State.size() + 1), "H5Tset_size");
    detail::ensure_valid(H5Tset_strpad(RngAttrTypeId.get(), H5T_STR_NULLTERM), "H5Tset_strpad");
    detail::H5Attr RngAttrId(H5Acreate2(DataSetId.get(), schema::lattice::rng_state_attribute.data(),
                                        RngAttrTypeId.get(), RngAttrSpaceId.get(), H5P_DEFAULT, H5P_DEFAULT));
    detail::ensure_valid(RngAttrId.get(), "H5Acreate2(RngState)");
    detail::ensure_valid(H5Awrite(RngAttrId.get(), RngAttrTypeId.get(), State.c_str()), "H5Awrite(RngState)");
}

template <typename T>
inline void Hdf5StorageBackend::readLattice(const fs::path& FileName, const std::string& LatticeID, Lattice<T>& field,
                                            std::stringstream& RngState) {
    detail::OmpLockGuard Lock(&_IoLock);
    detail::H5File       FileId(H5Fopen(FileName.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT));
    detail::ensure_valid(FileId.get(), "H5Fopen");
    detail::H5Dataset DataSetId(H5Dopen2(FileId.get(), LatticeID.c_str(), H5P_DEFAULT));
    detail::ensure_valid(DataSetId.get(), "H5Dopen2");
    detail::H5Dataspace DataSpaceId(H5Dget_space(DataSetId.get()));
    detail::ensure_valid(DataSpaceId.get(), "H5Dget_space");
    const int DataSpaceNDims = H5Sget_simple_extent_ndims(DataSpaceId.get());
    if (DataSpaceNDims != 1) {
        throw std::runtime_error("Hdf5StorageBackend - Error while reading configuration (dataspace mismatch)");
    }
    std::vector<hsize_t> Entries(DataSpaceNDims);
    detail::ensure_valid(H5Sget_simple_extent_dims(DataSpaceId.get(), Entries.data(), nullptr),
                         "H5Sget_simple_extent_dims");
    if (Entries[0] != field.size()) {
        throw std::runtime_error("Hdf5StorageBackend - Error while reading configuration (dimension mismatch)");
    }
    detail::H5Attr SizeAttrId(H5Aopen(DataSetId.get(), schema::lattice::field_sizes_attribute.data(), H5P_DEFAULT));
    detail::ensure_valid(SizeAttrId.get(), "H5Aopen(sizes)");
    detail::H5Dataspace SizeAttrSpaceId(H5Aget_space(SizeAttrId.get()));
    detail::ensure_valid(SizeAttrSpaceId.get(), "H5Aget_space(sizes)");
    const int            SizeAttrNDims = H5Sget_simple_extent_ndims(SizeAttrSpaceId.get());
    std::vector<hsize_t> SizeAttrDims(SizeAttrNDims, 0);
    detail::ensure_valid(H5Sget_simple_extent_dims(SizeAttrSpaceId.get(), SizeAttrDims.data(), nullptr),
                         "H5Sget_simple_extent_dims(sizes attr)");
    if (SizeAttrDims[0] != field.getDim()) {
        throw std::runtime_error("Hdf5StorageBackend - Error while reading configuration (dimension mismatch)");
    }
    std::vector<Indexing::size_type> Sizes(SizeAttrDims[0], 0);
    detail::H5Type                   SizeAttrTypeId(H5Aget_type(SizeAttrId.get()), true);
    detail::ensure_valid(SizeAttrTypeId.get(), "H5Aget_type(sizes)");
    auto SizeAttrTypeIdCheck = detail::make_type<typename Lattice<T>::size_type>();
    if (H5Tequal(SizeAttrTypeId.get(), SizeAttrTypeIdCheck.get()) != 1) {
        throw std::runtime_error(
            "Hdf5StorageBackend - Error while reading configuration (sizes attribute type mismatch)");
    }
    detail::ensure_valid(H5Aread(SizeAttrId.get(), SizeAttrTypeId.get(), Sizes.data()), "H5Aread(sizes)");
    if (Sizes != field.getSizes()) {
        throw std::runtime_error("Hdf5StorageBackend - Error while reading configuration (dimension mismatch)");
    }
    detail::H5Type DataTypeId(H5Dget_type(DataSetId.get()), true);
    detail::ensure_valid(DataTypeId.get(), "H5Dget_type");
    auto DataTypeIdCheck = detail::make_type<T>();
    if (H5Tequal(DataTypeId.get(), DataTypeIdCheck.get()) != 1) {
        throw std::runtime_error("Hdf5StorageBackend - Error while reading configuration (field type mismatch)");
    }
    detail::ensure_valid(
        H5Dread(DataSetId.get(), DataTypeId.get(), DataSpaceId.get(), DataSpaceId.get(), H5P_DEFAULT, field.data()),
        "H5Dread");
    detail::H5Attr RngAttrId(H5Aopen(DataSetId.get(), schema::lattice::rng_state_attribute.data(), H5P_DEFAULT));
    detail::ensure_valid(RngAttrId.get(), "H5Aopen(RngState)");
    hsize_t           RngAttrSize = H5Aget_storage_size(RngAttrId.get());
    std::vector<char> RngRaw(RngAttrSize + 1);
    detail::H5Type    RngAttrType(H5Aget_type(RngAttrId.get()), true);
    detail::ensure_valid(RngAttrType.get(), "H5Aget_type(RngState)");
    detail::ensure_valid(H5Aread(RngAttrId.get(), RngAttrType.get(), RngRaw.data()), "H5Aread(RngState)");
    RngState.str(std::string());
    RngState << std::string(RngRaw.begin(), RngRaw.end());
}

}  // namespace reticolo::storage
