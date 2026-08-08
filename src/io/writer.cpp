#include <reticolo/cli/parser.hpp>
#include <reticolo/core/field/lattice.hpp>
#include <reticolo/core/field/matrix_link_lattice.hpp>
#include <reticolo/core/log/log.hpp>
#include <reticolo/core/rng/fast_rng.hpp>
#include <reticolo/io/writer.hpp>

#include "hid_guard.hpp"

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

// The git stamp is regenerated at BUILD time (cmake/ReticoloBuildStamp.cmake)
// into this header, so `/run@commit` is the commit that produced THIS binary —
// not whatever HEAD was at the last `cmake` invocation. This TU is the only
// consumer, so a new commit recompiles one file rather than the whole tree.
// __has_include keeps the file compilable outside the CMake build (clang-tidy on
// a stale DB, an IDE indexer) — it falls back to the compile-definition value.
#if __has_include(<reticolo_build_stamp.h>)
    #include <reticolo_build_stamp.h>
#endif

#ifndef RETICOLO_VERSION
    #define RETICOLO_VERSION "0.0.0"
#endif
#ifdef RETICOLO_STAMP_GIT_COMMIT
    #define RETICOLO_RUN_COMMIT RETICOLO_STAMP_GIT_COMMIT
#elif defined(RETICOLO_GIT_COMMIT)
    #define RETICOLO_RUN_COMMIT RETICOLO_GIT_COMMIT
#else
    #define RETICOLO_RUN_COMMIT "unknown"
#endif
#ifdef RETICOLO_STAMP_GIT_BRANCH
    #define RETICOLO_RUN_BRANCH RETICOLO_STAMP_GIT_BRANCH
#elif defined(RETICOLO_GIT_BRANCH)
    #define RETICOLO_RUN_BRANCH RETICOLO_GIT_BRANCH
#else
    #define RETICOLO_RUN_BRANCH "unknown"
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
        hid_t const cid = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<float>));
        H5Tinsert(cid, "r", 0, H5T_NATIVE_FLOAT);
        H5Tinsert(cid, "i", sizeof(float), H5T_NATIVE_FLOAT);
        return cid;
    }();
    return t;
}

template <>
hid_t native_type<std::complex<double>>() {
    static hid_t const t = []() {
        hid_t const cid = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<double>));
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
// handle on the parent of the leaf. Ownership travels with the returned guard —
// this used to hand back a bare hid_t under a "caller must H5Gclose()" comment.
detail::GroupId ensure_parent_groups(hid_t file, std::vector<std::string> const& segments) {
    detail::GroupId current{H5Gopen2(file, "/", H5P_DEFAULT)};
    hid_check(current.get(), "ensure_parent_groups Gopen2 /");

    for (std::size_t i = 0; i + 1 < segments.size(); ++i) {
        std::string const& name = segments[i];
        htri_t const exists     = H5Lexists(current.get(), name.c_str(), H5P_DEFAULT);
        detail::GroupId next{
            exists > 0
                ? H5Gopen2(current.get(), name.c_str(), H5P_DEFAULT)
                : H5Gcreate2(current.get(), name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)};
        hid_check(next.get(), "ensure_parent_groups segment");
        current = std::move(next);
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
    (void)std::strftime(buf.data(), buf.size(), "%Y-%m-%dT%H:%M:%SZ", &tm);
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
        return {buf.data()};
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
    detail::TypeId const dtype{H5Tcopy(H5T_C_S1)};
    hid_check(dtype.get(), "Tcopy");
    herr_check(H5Tset_size(dtype.get(), H5T_VARIABLE), "Tset_size");
    herr_check(H5Tset_strpad(dtype.get(), H5T_STR_NULLTERM), "Tset_strpad");
    detail::SpaceId const space{H5Screate(H5S_SCALAR)};
    hid_check(space.get(), "Screate scalar");
    detail::AttrId const attr{
        H5Acreate2(obj, name, dtype.get(), space.get(), H5P_DEFAULT, H5P_DEFAULT)};
    hid_check(attr.get(), "Acreate2");
    char const* cstr = value.c_str();
    // HDF5 takes the address of the char* for a variable-length string write.
    // NOLINTNEXTLINE(bugprone-multi-level-implicit-pointer-conversion)
    herr_check(H5Awrite(attr.get(), dtype.get(), &cstr), "Awrite string");
}

template <class T>
void write_scalar_attr(hid_t obj, char const* name, T const& value) {
    detail::SpaceId const space{H5Screate(H5S_SCALAR)};
    hid_check(space.get(), "Screate scalar");
    detail::AttrId const attr{
        H5Acreate2(obj, name, native_type<T>(), space.get(), H5P_DEFAULT, H5P_DEFAULT)};
    hid_check(attr.get(), "Acreate2");
    herr_check(H5Awrite(attr.get(), native_type<T>(), &value), "Awrite scalar");
}

}  // namespace

// Writer::Impl
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

    void stamp_run(int argc, char const* const* argv) const {
        detail::GroupId const grp{H5Gcreate2(file, "/run", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)};
        hid_check(grp.get(), "stamp_run create /run");

        write_string_attr(grp.get(), "cmdline", join_argv(argc, argv));
        write_string_attr(grp.get(), "version", RETICOLO_VERSION);
        write_string_attr(grp.get(), "commit", RETICOLO_RUN_COMMIT);
        write_string_attr(grp.get(), "build_type", RETICOLO_BUILD_TYPE);
        write_string_attr(grp.get(), "compile_flags", RETICOLO_COMPILE_FLAGS);
        write_string_attr(grp.get(), "hostname", hostname_str());
        write_string_attr(grp.get(), "started_utc", utc_now_iso());
        write_string_attr(grp.get(), "hdf5_library_version", hdf5_library_version_str());
        write_string_attr(grp.get(), "hdf5_complex_schema", "legacy_compound");
    }
};

// Series<T>::Impl
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
        } catch (...) {  // NOLINT(bugprone-empty-catch) — a destructor must not propagate
            // dtor cannot throw
        }
        if (dataset != H5I_INVALID_HID) {
            H5Dclose(dataset);
        }
    }

    void append(T const& v) {
        std::scoped_lock const lock{*mu_ptr};
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

        detail::SpaceId const fspace{H5Dget_space(dataset)};
        hid_check(fspace.get(), "Dget_space");
        hsize_t const offset = total_written;
        herr_check(
            H5Sselect_hyperslab(fspace.get(), H5S_SELECT_SET, &offset, nullptr, &count, nullptr),
            "Sselect_hyperslab");

        detail::SpaceId const mspace{H5Screate_simple(1, &count, nullptr)};
        hid_check(mspace.get(), "Screate_simple");

        herr_check(
            H5Dwrite(
                dataset, native_type<T>(), mspace.get(), fspace.get(), H5P_DEFAULT, buffer.data()),
            "Dwrite");

        total_written = new_size;
        buffer.clear();
    }
};

// Series<T> definitions
template <class T>
Series<T>::Series() = default;

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
    std::scoped_lock const lock{*impl_->mu_ptr};
    impl_->flush_locked();
}

template <class T>
std::size_t Series<T>::size() const noexcept {
    if (!impl_) {
        return 0;
    }
    std::scoped_lock const lock{*impl_->mu_ptr};
    return impl_->total_written + impl_->buffer.size();
}

template <class T>
bool Series<T>::valid() const noexcept {
    return impl_ != nullptr;
}

// Writer definitions
Writer::Writer(std::filesystem::path const& path,
               int argc,
               char const* const* argv,
               cli::Parser const* params)
    : impl_{std::make_unique<Impl>(path, argc, argv)} {
    if (params != nullptr) {
        params->stamp(*this);
    }
    log::info("io", "opened {}", path.string());
}

Writer::~Writer()                            = default;
Writer::Writer(Writer&&) noexcept            = default;
Writer& Writer::operator=(Writer&&) noexcept = default;

std::filesystem::path const& Writer::path() const noexcept {
    return impl_->file_path;
}

void Writer::start_phase(std::string_view phase) {
    std::scoped_lock const lock{impl_->mu};
    if (std::ranges::find(impl_->phases, phase) != impl_->phases.end()) {
        throw std::runtime_error{"Writer::start_phase: phase '" + std::string(phase) +
                                 "' already started"};
    }
    std::string const phase_path = "/" + std::string(phase);
    htri_t const exists          = H5Lexists(impl_->file, phase_path.c_str(), H5P_DEFAULT);
    if (exists > 0) {
        throw std::runtime_error{"Writer::start_phase: '" + phase_path + "' already exists"};
    }
    detail::GroupId const grp{
        H5Gcreate2(impl_->file, phase_path.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)};
    hid_check(grp.get(), "start_phase create");
    impl_->phases.emplace_back(phase);
}

template <class T>
Series<T> Writer::series(std::string_view path, std::size_t chunk) {
    std::scoped_lock const lock{impl_->mu};

    auto segments = split_path(path);
    if (segments.empty()) {
        throw std::invalid_argument{"Writer::series: empty path"};
    }

    detail::GroupId const parent = ensure_parent_groups(impl_->file, segments);

    hsize_t const initial = 0;
    hsize_t const maxdim  = H5S_UNLIMITED;
    detail::SpaceId const space{H5Screate_simple(1, &initial, &maxdim)};
    hid_check(space.get(), "Screate_simple");

    detail::PropId const dcpl{H5Pcreate(H5P_DATASET_CREATE)};
    hsize_t const chunkdim = chunk == 0 ? hsize_t{4096} : static_cast<hsize_t>(chunk);
    H5Pset_chunk(dcpl.get(), 1, &chunkdim);

    // The ONLY id here that outlives this scope: Series<T>::Impl takes ownership
    // of the dataset and closes it in its own destructor, so it stays unguarded.
    hid_t const dset = H5Dcreate2(parent.get(),
                                  segments.back().c_str(),
                                  native_type<T>(),
                                  space.get(),
                                  H5P_DEFAULT,
                                  dcpl.get(),
                                  H5P_DEFAULT);
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
    std::scoped_lock const lock{impl_->mu};
    auto const [obj_path, attr_name] = split_attr(path);
    auto segments                    = split_path(obj_path);

    detail::ObjId parent{};
    if (segments.empty()) {
        parent = detail::ObjId{H5Gopen2(impl_->file, "/", H5P_DEFAULT)};
    } else {
        detail::GroupId const pre = ensure_parent_groups(impl_->file, segments);
        std::string const& leaf   = segments.back();
        htri_t const ex           = H5Lexists(pre.get(), leaf.c_str(), H5P_DEFAULT);
        parent                    = detail::ObjId{
            ex > 0 ? H5Oopen(pre.get(), leaf.c_str(), H5P_DEFAULT)
                   : H5Gcreate2(pre.get(), leaf.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)};
    }
    hid_check(parent.get(), "attr open parent");

    if constexpr (std::is_same_v<T, std::string>) {
        write_string_attr(parent.get(), attr_name.c_str(), value);
    } else {
        write_scalar_attr<T>(parent.get(), attr_name.c_str(), value);
    }
}

// Explicit instantiations
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage) — token-pasting instantiation list, not a constant
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

// ---------------------------------------------------------------------------
// Field / rng-state writes (config-snapshot surface).
// ---------------------------------------------------------------------------
namespace {

hid_t native_type_for(Writer::ScalarKind k) {
    switch (k) {
        case Writer::ScalarKind::f32:
            return native_type<float>();
        case Writer::ScalarKind::f64:
            return native_type<double>();
        case Writer::ScalarKind::c32:
            return native_type<std::complex<float>>();
        case Writer::ScalarKind::c64:
            return native_type<std::complex<double>>();
    }
    return H5I_INVALID_HID;
}

char const* scalar_type_name(Writer::ScalarKind k) {
    switch (k) {
        case Writer::ScalarKind::f32:
            return "float";
        case Writer::ScalarKind::f64:
            return "double";
        case Writer::ScalarKind::c32:
            return "complex<float>";
        case Writer::ScalarKind::c64:
            return "complex<double>";
    }
    return "?";
}

char const* field_kind_name(Writer::FieldKind k) {
    switch (k) {
        case Writer::FieldKind::scalar:
            return "scalar";
        case Writer::FieldKind::link:
            return "link";
        case Writer::FieldKind::matrix_link:
            return "matrix_link";
    }
    return "?";
}

std::string join_shape(std::vector<std::size_t> const& shape) {
    std::string out;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out += ',';
        }
        out += std::to_string(shape[i]);
    }
    return out;
}

void write_1d_dataset(
    hid_t parent, char const* leaf, hid_t htype, void const* data, hsize_t count) {
    detail::SpaceId const space{H5Screate_simple(1, &count, nullptr)};
    hid_check(space.get(), "Screate_simple field");
    detail::DsetId const dset{
        H5Dcreate2(parent, leaf, htype, space.get(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)};
    hid_check(dset.get(), "Dcreate2 field");
    herr_check(H5Dwrite(dset.get(), htype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data), "Dwrite field");
}

}  // namespace

void Writer::write_field_raw_(std::string_view path,
                              void const* data,
                              std::size_t n_elems,
                              ScalarKind scalar_kind,
                              FieldKind kind,
                              std::vector<std::size_t> const& shape,
                              std::size_t n_components,
                              char const* group_name) {
    std::scoped_lock const lock{impl_->mu};

    auto segments = split_path(path);
    if (segments.empty()) {
        throw std::invalid_argument{"Writer::field: empty path"};
    }

    detail::GroupId const parent = ensure_parent_groups(impl_->file, segments);

    hid_t const htype = native_type_for(scalar_kind);
    write_1d_dataset(parent.get(), segments.back().c_str(), htype, data, n_elems);

    detail::ObjId const dset{H5Oopen(parent.get(), segments.back().c_str(), H5P_DEFAULT)};
    hid_check(dset.get(), "field reopen for attrs");

    write_string_attr(dset.get(), "kind", field_kind_name(kind));
    write_string_attr(dset.get(), "scalar_type", scalar_type_name(scalar_kind));
    write_string_attr(dset.get(), "shape", join_shape(shape));
    write_scalar_attr<std::uint64_t>(dset.get(), "n_components", n_components);
    if (group_name != nullptr && group_name[0] != '\0') {
        write_string_attr(dset.get(), "group", group_name);
    }
}

void Writer::rng_state(std::string_view path, FastRng const& rng) {
    std::scoped_lock const lock{impl_->mu};

    auto segments = split_path(path);
    if (segments.empty()) {
        throw std::invalid_argument{"Writer::rng_state: empty path"};
    }

    detail::GroupId const parent = ensure_parent_groups(impl_->file, segments);

    auto const& s = rng.state();
    write_1d_dataset(parent.get(), segments.back().c_str(), H5T_NATIVE_UINT64, s.data(), s.size());

    detail::ObjId const dset{H5Oopen(parent.get(), segments.back().c_str(), H5P_DEFAULT)};
    hid_check(dset.get(), "rng_state reopen for attrs");

    write_string_attr(dset.get(), "kind", "FastRng");
    write_scalar_attr<unsigned int>(
        dset.get(), "has_cached_normal", rng.has_cached_normal() ? 1U : 0U);
    write_scalar_attr<double>(dset.get(), "cached_normal", rng.cached_normal());
}

void Writer::rng_streams(std::string_view path,
                         std::string_view kind,
                         std::vector<std::uint64_t> const& words,
                         std::size_t n_streams,
                         std::size_t n_words) {
    std::scoped_lock const lock{impl_->mu};

    auto segments = split_path(path);
    if (segments.empty()) {
        throw std::invalid_argument{"Writer::rng_streams: empty path"};
    }
    if (words.size() != (n_streams + 1) * n_words) {
        throw std::invalid_argument{"Writer::rng_streams: word count mismatch"};
    }

    detail::GroupId const parent = ensure_parent_groups(impl_->file, segments);
    write_1d_dataset(
        parent.get(), segments.back().c_str(), H5T_NATIVE_UINT64, words.data(), words.size());

    detail::ObjId const dset{H5Oopen(parent.get(), segments.back().c_str(), H5P_DEFAULT)};
    hid_check(dset.get(), "rng_streams reopen for attrs");

    write_string_attr(dset.get(), "kind", std::string{kind});
    write_scalar_attr<std::uint64_t>(dset.get(), "n_streams", n_streams);
    write_scalar_attr<std::uint64_t>(dset.get(), "n_words", n_words);
}

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage) — token-pasting instantiation list, not a constant
#define RETICOLO_IO_INSTANTIATE_FIELD(T)                                                           \
    template void Writer::field<T>(std::string_view, Lattice<T> const&);
RETICOLO_IO_INSTANTIATE_FIELD(float)
RETICOLO_IO_INSTANTIATE_FIELD(double)
RETICOLO_IO_INSTANTIATE_FIELD(std::complex<float>)
RETICOLO_IO_INSTANTIATE_FIELD(std::complex<double>)
#undef RETICOLO_IO_INSTANTIATE_FIELD

}  // namespace reticolo::io
