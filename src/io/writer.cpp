#include <reticolo/io/writer.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <hdf5.h>

#if defined(_WIN32)
    #include <winsock2.h>
#else
    #include <unistd.h>
#endif

#ifndef RETICOLO_VERSION
    #define RETICOLO_VERSION "0.0.0"
#endif
#ifndef RETICOLO_GIT_COMMIT
    #define RETICOLO_GIT_COMMIT "unknown"
#endif
#ifndef RETICOLO_BUILD_TYPE
    #define RETICOLO_BUILD_TYPE "Unknown"
#endif
#ifndef RETICOLO_COMPILE_FLAGS
    #define RETICOLO_COMPILE_FLAGS ""
#endif

namespace reticolo::io {

namespace {

[[noreturn]] void hdf5_throw(char const* op) {
    throw std::runtime_error{std::string{"reticolo::io: HDF5 failure in "} + op};
}

void hid_check(hid_t h, char const* op) {
    if (h < 0) {
        hdf5_throw(op);
    }
}

void herr_check(herr_t e, char const* op) {
    if (e < 0) {
        hdf5_throw(op);
    }
}

// ---------------------------------------------------------------------------
// HDF5 native type lookup. For complex we build a compound {r, i} once.
// ---------------------------------------------------------------------------
template <class T>
hid_t native_type();

// clang-format off
template <> hid_t native_type<float>()              { return H5T_NATIVE_FLOAT; }
template <> hid_t native_type<double>()             { return H5T_NATIVE_DOUBLE; }
template <> hid_t native_type<int>()                { return H5T_NATIVE_INT; }
template <> hid_t native_type<long>()               { return H5T_NATIVE_LONG; }
template <> hid_t native_type<long long>()          { return H5T_NATIVE_LLONG; }
template <> hid_t native_type<unsigned int>()       { return H5T_NATIVE_UINT; }
template <> hid_t native_type<unsigned long>()      { return H5T_NATIVE_ULONG; }
template <> hid_t native_type<unsigned long long>() { return H5T_NATIVE_ULLONG; }
// clang-format on

template <>
hid_t native_type<std::complex<float>>() {
    static hid_t const t = []() {
        hid_t cid = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<float>));
        H5Tinsert(cid, "r", 0, H5T_NATIVE_FLOAT);
        H5Tinsert(cid, "i", sizeof(float), H5T_NATIVE_FLOAT);
        return cid;
    }();
    return t;
}

template <>
hid_t native_type<std::complex<double>>() {
    static hid_t const t = []() {
        hid_t cid = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<double>));
        H5Tinsert(cid, "r", 0, H5T_NATIVE_DOUBLE);
        H5Tinsert(cid, "i", sizeof(double), H5T_NATIVE_DOUBLE);
        return cid;
    }();
    return t;
}

// ---------------------------------------------------------------------------
// Path utilities. "/a/b/c" -> ["a","b","c"]; "/g/d@attr" -> ("/g/d","attr").
// ---------------------------------------------------------------------------
std::vector<std::string> split_path(std::string_view path) {
    std::vector<std::string> out;
    std::string cur;
    for (char const c : path) {
        if (c == '/') {
            if (!cur.empty()) {
                out.push_back(std::move(cur));
                cur.clear();
            }
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) {
        out.push_back(std::move(cur));
    }
    return out;
}

std::pair<std::string, std::string> split_attr(std::string_view path) {
    auto const pos = path.find('@');
    if (pos == std::string_view::npos) {
        throw std::invalid_argument{"reticolo::io: attr path must contain '@': '" +
                                    std::string(path) + "'"};
    }
    return {std::string(path.substr(0, pos)), std::string(path.substr(pos + 1))};
}

// Create intermediate groups (all but the last segment) and return an open
// handle on the parent of the leaf. Caller must H5Gclose() the returned hid.
hid_t ensure_parent_groups(hid_t file, std::vector<std::string> const& segments) {
    hid_t current = H5Gopen2(file, "/", H5P_DEFAULT);
    hid_check(current, "ensure_parent_groups Gopen2 /");

    if (segments.size() <= 1) {
        return current;
    }

    for (std::size_t i = 0; i + 1 < segments.size(); ++i) {
        std::string const& name = segments[i];
        htri_t const exists     = H5Lexists(current, name.c_str(), H5P_DEFAULT);
        hid_t next              = H5I_INVALID_HID;
        if (exists > 0) {
            next = H5Gopen2(current, name.c_str(), H5P_DEFAULT);
        } else {
            next = H5Gcreate2(current, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        }
        H5Gclose(current);
        hid_check(next, "ensure_parent_groups segment");
        current = next;
    }
    return current;
}

// ---------------------------------------------------------------------------
// Run metadata helpers.
// ---------------------------------------------------------------------------
std::string utc_now_iso() {
    auto const tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &tt);
#else
    gmtime_r(&tt, &tm);
#endif
    std::array<char, 32> buf{};
    std::strftime(buf.data(), buf.size(), "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf.data();
}

std::string hostname_str() {
    std::array<char, 256> buf{};
#if defined(_WIN32)
    DWORD sz = static_cast<DWORD>(buf.size());
    if (GetComputerNameA(buf.data(), &sz) != 0) {
        return std::string(buf.data(), sz);
    }
#else
    if (gethostname(buf.data(), buf.size()) == 0) {
        buf.back() = '\0';
        return std::string(buf.data());
    }
#endif
    return "<unknown>";
}

std::string hdf5_library_version_str() {
    unsigned maj = 0;
    unsigned min = 0;
    unsigned rel = 0;
    H5get_libversion(&maj, &min, &rel);
    return std::to_string(maj) + "." + std::to_string(min) + "." + std::to_string(rel);
}

std::string join_argv(int argc, char const* const* argv) {
    if (argc <= 0 || argv == nullptr) {
        return {};
    }
    std::string out;
    for (int i = 0; i < argc; ++i) {
        if (i > 0) {
            out += ' ';
        }
        out += argv[i];
    }
    return out;
}

void write_string_attr(hid_t obj, char const* name, std::string const& value) {
    hid_t dtype = H5Tcopy(H5T_C_S1);
    hid_check(dtype, "Tcopy");
    herr_check(H5Tset_size(dtype, H5T_VARIABLE), "Tset_size");
    herr_check(H5Tset_strpad(dtype, H5T_STR_NULLTERM), "Tset_strpad");
    hid_t space = H5Screate(H5S_SCALAR);
    hid_check(space, "Screate scalar");
    hid_t attr = H5Acreate2(obj, name, dtype, space, H5P_DEFAULT, H5P_DEFAULT);
    hid_check(attr, "Acreate2");
    char const* cstr = value.c_str();
    herr_check(H5Awrite(attr, dtype, &cstr), "Awrite string");
    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(dtype);
}

template <class T>
void write_scalar_attr(hid_t obj, char const* name, T const& value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_check(space, "Screate scalar");
    hid_t attr = H5Acreate2(obj, name, native_type<T>(), space, H5P_DEFAULT, H5P_DEFAULT);
    hid_check(attr, "Acreate2");
    herr_check(H5Awrite(attr, native_type<T>(), &value), "Awrite scalar");
    H5Aclose(attr);
    H5Sclose(space);
}

}  // namespace

// ============================================================================
//  Writer::Impl
// ============================================================================
struct Writer::Impl {
    std::filesystem::path file_path;
    hid_t file = H5I_INVALID_HID;
    std::mutex mu;
    std::vector<std::string> phases;

    Impl(std::filesystem::path const& p, int argc, char const* const* argv) : file_path{p} {
        // Silence HDF5's default stderr error printer; we throw on failure and
        // emit our own diagnostic strings.
        H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

        file = H5Fcreate(p.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (file < 0) {
            throw std::runtime_error{"reticolo::io::Writer: cannot create '" + p.string() + "'"};
        }
        stamp_run(argc, argv);
    }

    ~Impl() noexcept {
        if (file != H5I_INVALID_HID) {
            H5Fflush(file, H5F_SCOPE_GLOBAL);
            H5Fclose(file);
        }
    }

    Impl(Impl const&)            = delete;
    Impl& operator=(Impl const&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;

    void stamp_run(int argc, char const* const* argv) {
        hid_t grp = H5Gcreate2(file, "/run", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_check(grp, "stamp_run create /run");

        write_string_attr(grp, "cmdline", join_argv(argc, argv));
        write_string_attr(grp, "version", RETICOLO_VERSION);
        write_string_attr(grp, "commit", RETICOLO_GIT_COMMIT);
        write_string_attr(grp, "build_type", RETICOLO_BUILD_TYPE);
        write_string_attr(grp, "compile_flags", RETICOLO_COMPILE_FLAGS);
        write_string_attr(grp, "hostname", hostname_str());
        write_string_attr(grp, "started_utc", utc_now_iso());
        write_string_attr(grp, "hdf5_library_version", hdf5_library_version_str());
        write_string_attr(grp, "hdf5_complex_schema", "legacy_compound");

        H5Gclose(grp);
    }
};

// ============================================================================
//  Series<T>::Impl
// ============================================================================
template <class T>
struct Series<T>::Impl {
    std::vector<T> buffer;
    std::size_t chunk_rows    = 0;
    std::size_t total_written = 0;
    hid_t dataset             = H5I_INVALID_HID;
    std::mutex* mu_ptr        = nullptr;  // borrowed from Writer

    Impl()                           = default;
    Impl(Impl const&)                = delete;
    Impl& operator=(Impl const&)     = delete;
    Impl(Impl&&) noexcept            = default;
    Impl& operator=(Impl&&) noexcept = default;

    ~Impl() {
        try {
            flush_locked();
        } catch (...) {
            // dtor cannot throw
        }
        if (dataset != H5I_INVALID_HID) {
            H5Dclose(dataset);
        }
    }

    void append(T const& v) {
        std::lock_guard<std::mutex> lock{*mu_ptr};
        buffer.push_back(v);
        if (buffer.size() >= chunk_rows) {
            flush_locked();
        }
    }

    void flush_locked() {
        if (buffer.empty() || dataset == H5I_INVALID_HID) {
            return;
        }
        hsize_t const count    = buffer.size();
        hsize_t const new_size = total_written + count;
        herr_check(H5Dset_extent(dataset, &new_size), "Dset_extent");

        hid_t fspace = H5Dget_space(dataset);
        hid_check(fspace, "Dget_space");
        hsize_t offset = total_written;
        herr_check(H5Sselect_hyperslab(fspace, H5S_SELECT_SET, &offset, nullptr, &count, nullptr),
                   "Sselect_hyperslab");

        hid_t mspace = H5Screate_simple(1, &count, nullptr);
        if (mspace < 0) {
            H5Sclose(fspace);
            hdf5_throw("Screate_simple");
        }

        herr_t const e =
            H5Dwrite(dataset, native_type<T>(), mspace, fspace, H5P_DEFAULT, buffer.data());
        H5Sclose(mspace);
        H5Sclose(fspace);
        herr_check(e, "Dwrite");

        total_written = new_size;
        buffer.clear();
    }
};

// ============================================================================
//  Series<T> definitions
// ============================================================================
template <class T>
Series<T>::~Series() = default;

template <class T>
Series<T>::Series(Series&&) noexcept = default;

template <class T>
Series<T>& Series<T>::operator=(Series&&) noexcept = default;

template <class T>
Series<T>::Series(std::unique_ptr<Impl> impl) noexcept : impl_{std::move(impl)} {}

template <class T>
void Series<T>::append(T const& value) {
    if (impl_) {
        impl_->append(value);
    }
}

template <class T>
void Series<T>::flush() {
    if (!impl_) {
        return;
    }
    std::lock_guard<std::mutex> lock{*impl_->mu_ptr};
    impl_->flush_locked();
}

template <class T>
std::size_t Series<T>::size() const noexcept {
    if (!impl_) {
        return 0;
    }
    std::lock_guard<std::mutex> lock{*impl_->mu_ptr};
    return impl_->total_written + impl_->buffer.size();
}

template <class T>
bool Series<T>::valid() const noexcept {
    return impl_ != nullptr;
}

// ============================================================================
//  Writer definitions
// ============================================================================
Writer::Writer(std::filesystem::path const& path, int argc, char const* const* argv)
    : impl_{std::make_unique<Impl>(path, argc, argv)} {}

Writer::~Writer()                            = default;
Writer::Writer(Writer&&) noexcept            = default;
Writer& Writer::operator=(Writer&&) noexcept = default;

std::filesystem::path const& Writer::path() const noexcept {
    return impl_->file_path;
}

void Writer::start_phase(std::string_view phase) {
    std::lock_guard<std::mutex> lock{impl_->mu};
    if (std::ranges::find(impl_->phases, phase) != impl_->phases.end()) {
        throw std::runtime_error{"Writer::start_phase: phase '" + std::string(phase) +
                                 "' already started"};
    }
    std::string const phase_path = "/" + std::string(phase);
    htri_t const exists          = H5Lexists(impl_->file, phase_path.c_str(), H5P_DEFAULT);
    if (exists > 0) {
        throw std::runtime_error{"Writer::start_phase: '" + phase_path + "' already exists"};
    }
    hid_t grp = H5Gcreate2(impl_->file, phase_path.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_check(grp, "start_phase create");
    H5Gclose(grp);
    impl_->phases.emplace_back(phase);
}

template <class T>
Series<T> Writer::series(std::string_view path, std::size_t chunk) {
    std::lock_guard<std::mutex> lock{impl_->mu};

    auto segments = split_path(path);
    if (segments.empty()) {
        throw std::invalid_argument{"Writer::series: empty path"};
    }

    hid_t parent = ensure_parent_groups(impl_->file, segments);

    hsize_t initial = 0;
    hsize_t maxdim  = H5S_UNLIMITED;
    hid_t space     = H5Screate_simple(1, &initial, &maxdim);
    hid_check(space, "Screate_simple");

    hid_t dcpl       = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t chunkdim = chunk == 0 ? hsize_t{4096} : static_cast<hsize_t>(chunk);
    H5Pset_chunk(dcpl, 1, &chunkdim);

    hid_t dset = H5Dcreate2(
        parent, segments.back().c_str(), native_type<T>(), space, H5P_DEFAULT, dcpl, H5P_DEFAULT);

    H5Pclose(dcpl);
    H5Sclose(space);
    H5Gclose(parent);

    hid_check(dset, "Dcreate2");

    auto impl        = std::make_unique<typename Series<T>::Impl>();
    impl->chunk_rows = chunk == 0 ? 4096 : chunk;
    impl->dataset    = dset;
    impl->mu_ptr     = &impl_->mu;
    impl->buffer.reserve(impl->chunk_rows);
    return Series<T>{std::move(impl)};
}

template <class T>
void Writer::attr(std::string_view path, T const& value) {
    std::lock_guard<std::mutex> lock{impl_->mu};
    auto const [obj_path, attr_name] = split_attr(path);
    auto segments                    = split_path(obj_path);

    hid_t parent = H5I_INVALID_HID;
    if (segments.empty()) {
        parent = H5Gopen2(impl_->file, "/", H5P_DEFAULT);
    } else {
        hid_t pre               = ensure_parent_groups(impl_->file, segments);
        std::string const& leaf = segments.back();
        htri_t const ex         = H5Lexists(pre, leaf.c_str(), H5P_DEFAULT);
        hid_t leaf_grp          = H5I_INVALID_HID;
        if (ex > 0) {
            leaf_grp = H5Oopen(pre, leaf.c_str(), H5P_DEFAULT);
        } else {
            leaf_grp = H5Gcreate2(pre, leaf.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        }
        H5Gclose(pre);
        parent = leaf_grp;
    }
    hid_check(parent, "attr open parent");

    if constexpr (std::is_same_v<T, std::string>) {
        write_string_attr(parent, attr_name.c_str(), value);
    } else {
        write_scalar_attr<T>(parent, attr_name.c_str(), value);
    }

    H5Oclose(parent);
}

// ============================================================================
//  Explicit instantiations
// ============================================================================
#define RETICOLO_IO_INSTANTIATE(T)                                                                 \
    template class Series<T>;                                                                      \
    template Series<T> Writer::series<T>(std::string_view, std::size_t);                           \
    template void Writer::attr<T>(std::string_view, T const&);

RETICOLO_IO_INSTANTIATE(float)
RETICOLO_IO_INSTANTIATE(double)
RETICOLO_IO_INSTANTIATE(int)
RETICOLO_IO_INSTANTIATE(long)
RETICOLO_IO_INSTANTIATE(long long)
RETICOLO_IO_INSTANTIATE(unsigned int)
RETICOLO_IO_INSTANTIATE(unsigned long)
RETICOLO_IO_INSTANTIATE(unsigned long long)
RETICOLO_IO_INSTANTIATE(std::complex<float>)
RETICOLO_IO_INSTANTIATE(std::complex<double>)

#undef RETICOLO_IO_INSTANTIATE

template void Writer::attr<std::string>(std::string_view, std::string const&);

}  // namespace reticolo::io
