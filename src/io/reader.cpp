#include <reticolo/core/log/log.hpp>
#include <reticolo/io/reader.hpp>

#include "hid_guard.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <hdf5.h>

namespace reticolo::io {

namespace {

[[noreturn]] void hdf5_throw(char const* op) {
    throw std::runtime_error{std::string{"reticolo::io::Reader: HDF5 failure in "} + op};
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

hid_t native_type_for(Writer::ScalarKind k) {
    switch (k) {
        case Writer::ScalarKind::f32:
            return H5T_NATIVE_FLOAT;
        case Writer::ScalarKind::f64:
            return H5T_NATIVE_DOUBLE;
        case Writer::ScalarKind::c32: {
            static hid_t const t = []() {
                hid_t const cid = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<float>));
                H5Tinsert(cid, "r", 0, H5T_NATIVE_FLOAT);
                H5Tinsert(cid, "i", sizeof(float), H5T_NATIVE_FLOAT);
                return cid;
            }();
            return t;
        }
        case Writer::ScalarKind::c64: {
            static hid_t const t = []() {
                hid_t const cid = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<double>));
                H5Tinsert(cid, "r", 0, H5T_NATIVE_DOUBLE);
                H5Tinsert(cid, "i", sizeof(double), H5T_NATIVE_DOUBLE);
                return cid;
            }();
            return t;
        }
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

std::pair<std::string, std::string> split_attr(std::string_view path) {
    auto const pos = path.find('@');
    if (pos == std::string_view::npos) {
        throw std::invalid_argument{"reticolo::io::Reader: attr path must contain '@': '" +
                                    std::string(path) + "'"};
    }
    return {std::string(path.substr(0, pos)), std::string(path.substr(pos + 1))};
}

std::string read_string_attr(hid_t obj, char const* name) {
    detail::AttrId const attr{H5Aopen(obj, name, H5P_DEFAULT)};
    hid_check(attr.get(), "Aopen string");
    detail::TypeId const dtype{H5Aget_type(attr.get())};
    hid_check(dtype.get(), "Aget_type");
    if (H5Tis_variable_str(dtype.get()) > 0) {
        char* cstr = nullptr;
        // HDF5 takes the address of the char* for a variable-length string read.
        // NOLINTNEXTLINE(bugprone-multi-level-implicit-pointer-conversion)
        herr_t const e1 = H5Aread(attr.get(), dtype.get(), &cstr);
        std::string out;
        if (e1 >= 0 && cstr != nullptr) {
            out = cstr;
            // H5Aget_space mints an id of its own — reclaim needs it, and it must
            // be closed like any other. Reading it inline (as this once did) leaks
            // one dataspace per vlen attribute read, on the SUCCESS path.
            detail::SpaceId const aspace{H5Aget_space(attr.get())};
            // NOLINTNEXTLINE(bugprone-multi-level-implicit-pointer-conversion)
            H5Dvlen_reclaim(dtype.get(), aspace.get(), H5P_DEFAULT, &cstr);
        }
        if (e1 < 0) {
            hdf5_throw("Aread vlen string");
        }
        return out;
    }
    // Fixed-length string fallback.
    auto const sz = H5Tget_size(dtype.get());
    std::vector<char> buf(sz + 1, '\0');
    herr_check(H5Aread(attr.get(), dtype.get(), buf.data()), "Aread fixed string");
    return std::string{buf.data()};
}

template <class T>
T read_scalar_attr(hid_t obj, char const* name) {
    detail::AttrId const attr{H5Aopen(obj, name, H5P_DEFAULT)};
    hid_check(attr.get(), "Aopen scalar");
    T value{};
    herr_check(H5Aread(attr.get(), native_type<T>(), &value), "Aread scalar");
    return value;
}

std::vector<std::size_t> parse_shape_string(std::string const& s) {
    std::vector<std::size_t> out;
    std::size_t pos = 0;
    while (pos < s.size()) {
        std::size_t end = s.find(',', pos);
        if (end == std::string::npos) {
            end = s.size();
        }
        if (end > pos) {
            out.push_back(static_cast<std::size_t>(std::stoull(s.substr(pos, end - pos))));
        }
        pos = end + 1;
    }
    return out;
}

}  // namespace

struct Reader::Impl {
    std::filesystem::path file_path;
    hid_t file = H5I_INVALID_HID;

    explicit Impl(std::filesystem::path const& p) : file_path{p} {
        H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
        file = H5Fopen(p.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file < 0) {
            throw std::runtime_error{"reticolo::io::Reader: cannot open '" + p.string() + "'"};
        }
    }

    ~Impl() noexcept {
        if (file != H5I_INVALID_HID) {
            H5Fclose(file);
        }
    }

    Impl(Impl const&)            = delete;
    Impl& operator=(Impl const&) = delete;
    Impl(Impl&&)                 = delete;
    Impl& operator=(Impl&&)      = delete;
};

Reader::Reader(std::filesystem::path const& path) : impl_{std::make_unique<Impl>(path)} {
    log::info("io", "read   {}", path.string());
}

Reader::~Reader()                            = default;
Reader::Reader(Reader&&) noexcept            = default;
Reader& Reader::operator=(Reader&&) noexcept = default;

std::filesystem::path const& Reader::path() const noexcept {
    return impl_->file_path;
}

bool Reader::has(std::string_view path) const {
    if (path.find('@') != std::string_view::npos) {
        auto const [obj_path, attr_name] = split_attr(path);
        std::string const probe          = obj_path.empty() ? std::string{"/"} : obj_path;
        htri_t const lex                 = H5Lexists(impl_->file, probe.c_str(), H5P_DEFAULT);
        if (lex <= 0) {
            return false;
        }
        detail::ObjId const obj{H5Oopen(impl_->file, probe.c_str(), H5P_DEFAULT)};
        if (!obj.valid()) {
            return false;
        }
        return H5Aexists(obj.get(), attr_name.c_str()) > 0;
    }
    std::string const p{path};
    htri_t const ex = H5Lexists(impl_->file, p.c_str(), H5P_DEFAULT);
    return ex > 0;
}

template <class T>
T Reader::attr(std::string_view path) const {
    auto const [obj_path, attr_name] = split_attr(path);
    std::string const open_path      = obj_path.empty() ? std::string{"/"} : obj_path;
    detail::ObjId const obj{H5Oopen(impl_->file, open_path.c_str(), H5P_DEFAULT)};
    hid_check(obj.get(), "attr Oopen");
    if constexpr (std::is_same_v<T, std::string>) {
        return read_string_attr(obj.get(), attr_name.c_str());
    } else {
        return read_scalar_attr<T>(obj.get(), attr_name.c_str());
    }
}

std::vector<std::size_t> Reader::field_shape(std::string_view path) const {
    std::string const p{path};
    detail::ObjId const dset{H5Oopen(impl_->file, p.c_str(), H5P_DEFAULT)};
    hid_check(dset.get(), "field_shape Oopen");
    return parse_shape_string(read_string_attr(dset.get(), "shape"));
}

void Reader::read_field_raw_(std::string_view path,
                             void* data_out,
                             std::size_t n_elems,
                             Writer::ScalarKind scalar_kind,
                             Writer::FieldKind kind,
                             std::span<std::size_t const> expected_shape,
                             std::size_t expected_n_components) const {
    std::string const p{path};
    detail::ObjId const dset{H5Oopen(impl_->file, p.c_str(), H5P_DEFAULT)};
    hid_check(dset.get(), "field Oopen");

    // No hand-rolled close here any more: `dset` unwinds with the scope, which is
    // also what makes the hid_check/herr_check throws below leak-free.
    auto fail = [&](std::string const& msg) {
        throw std::runtime_error{"reticolo::io::Reader::field('" + p + "'): " + msg};
    };

    std::string const file_kind = read_string_attr(dset.get(), "kind");
    if (file_kind != field_kind_name(kind)) {
        fail("kind mismatch: file='" + file_kind + "' expected='" + field_kind_name(kind) + "'");
    }
    std::string const file_scalar = read_string_attr(dset.get(), "scalar_type");
    if (file_scalar != scalar_type_name(scalar_kind)) {
        fail("scalar_type mismatch: file='" + file_scalar + "' expected='" +
             scalar_type_name(scalar_kind) + "'");
    }
    std::string const shape_s    = read_string_attr(dset.get(), "shape");
    auto const file_shape        = parse_shape_string(shape_s);
    auto const file_n_components = read_scalar_attr<std::uint64_t>(dset.get(), "n_components");
    // file_shape is a vector, expected_shape a span over the field's inline
    // extents — compare element-wise rather than by container type.
    if (!std::ranges::equal(file_shape, expected_shape)) {
        fail("shape mismatch: file='" + shape_s + "'");
    }
    if (file_n_components != expected_n_components) {
        fail("n_components mismatch: file=" + std::to_string(file_n_components) +
             " expected=" + std::to_string(expected_n_components));
    }

    detail::SpaceId const space{H5Dget_space(dset.get())};
    hid_check(space.get(), "field Dget_space");
    hsize_t dims[1] = {0};
    if (H5Sget_simple_extent_ndims(space.get()) != 1) {
        fail("dataset is not 1-D");
    }
    H5Sget_simple_extent_dims(space.get(), dims, nullptr);
    if (dims[0] != n_elems) {
        fail("element count mismatch: file=" + std::to_string(dims[0]) +
             " expected=" + std::to_string(n_elems));
    }

    herr_check(
        H5Dread(dset.get(), native_type_for(scalar_kind), H5S_ALL, H5S_ALL, H5P_DEFAULT, data_out),
        "field Dread");
}

FastRng Reader::rng_state(std::string_view path) const {
    std::string const p{path};
    detail::ObjId const dset{H5Oopen(impl_->file, p.c_str(), H5P_DEFAULT)};
    hid_check(dset.get(), "rng Oopen");

    auto fail = [&](std::string const& msg) {
        throw std::runtime_error{"reticolo::io::Reader::rng_state('" + p + "'): " + msg};
    };

    std::string const kind = read_string_attr(dset.get(), "kind");
    if (kind != "FastRng") {
        fail("kind mismatch: file='" + kind + "' expected='FastRng'");
    }
    auto const has_cached_u = read_scalar_attr<unsigned int>(dset.get(), "has_cached_normal");
    auto const cached       = read_scalar_attr<double>(dset.get(), "cached_normal");

    detail::SpaceId const space{H5Dget_space(dset.get())};
    hid_check(space.get(), "rng Dget_space");
    hsize_t dims[1] = {0};
    H5Sget_simple_extent_dims(space.get(), dims, nullptr);
    if (dims[0] != 4) {
        fail("expected 4 state words, file has " + std::to_string(dims[0]));
    }
    std::array<std::uint64_t, 4> s{};
    herr_check(H5Dread(dset.get(), H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, s.data()),
               "rng Dread");

    return FastRng::from_state(s, cached, has_cached_u != 0);
}

std::vector<std::uint64_t> Reader::rng_streams(std::string_view path,
                                               std::string_view expected_kind,
                                               std::size_t expected_n_streams,
                                               std::size_t expected_n_words) const {
    std::string const p{path};
    detail::ObjId const dset{H5Oopen(impl_->file, p.c_str(), H5P_DEFAULT)};
    hid_check(dset.get(), "rng_streams Oopen");

    auto fail = [&](std::string const& msg) {
        throw std::runtime_error{"reticolo::io::Reader::rng_streams('" + p + "'): " + msg};
    };

    std::string const kind = read_string_attr(dset.get(), "kind");
    if (kind != expected_kind) {
        fail("kind mismatch: file='" + kind + "' expected='" + std::string{expected_kind} + "'");
    }
    if (H5Aexists(dset.get(), "n_streams") <= 0) {
        fail("no n_streams attribute — single-generator layout? (use rng_state)");
    }
    auto const n_streams = read_scalar_attr<std::uint64_t>(dset.get(), "n_streams");
    auto const n_words   = read_scalar_attr<std::uint64_t>(dset.get(), "n_words");
    if (n_streams != expected_n_streams) {
        fail("n_streams mismatch: file=" + std::to_string(n_streams) +
             " expected=" + std::to_string(expected_n_streams) +
             " — a StreamSet resume must keep the checkpoint's stream count");
    }
    if (n_words != expected_n_words) {
        fail("n_words mismatch: file=" + std::to_string(n_words) +
             " expected=" + std::to_string(expected_n_words));
    }

    std::size_t const total = (n_streams + 1) * n_words;
    detail::SpaceId const space{H5Dget_space(dset.get())};
    hid_check(space.get(), "rng_streams Dget_space");
    hsize_t dims[1] = {0};
    H5Sget_simple_extent_dims(space.get(), dims, nullptr);
    if (dims[0] != total) {
        fail("word count mismatch: file=" + std::to_string(dims[0]) +
             " expected=" + std::to_string(total));
    }

    std::vector<std::uint64_t> words(total);
    herr_check(H5Dread(dset.get(), H5T_NATIVE_UINT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, words.data()),
               "rng_streams Dread");
    return words;
}

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage) — token-pasting instantiation list, not a constant
#define RETICOLO_IO_READER_INSTANTIATE(T) template T Reader::attr<T>(std::string_view) const;
RETICOLO_IO_READER_INSTANTIATE(float)
RETICOLO_IO_READER_INSTANTIATE(double)
RETICOLO_IO_READER_INSTANTIATE(int)
RETICOLO_IO_READER_INSTANTIATE(long)
RETICOLO_IO_READER_INSTANTIATE(long long)
RETICOLO_IO_READER_INSTANTIATE(unsigned int)
RETICOLO_IO_READER_INSTANTIATE(unsigned long)
RETICOLO_IO_READER_INSTANTIATE(unsigned long long)
#undef RETICOLO_IO_READER_INSTANTIATE
template std::string Reader::attr<std::string>(std::string_view) const;

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage) — token-pasting instantiation list, not a constant
#define RETICOLO_IO_READER_INSTANTIATE_FIELD(T)                                                    \
    template void Reader::field<T>(std::string_view, Lattice<T>&) const;
RETICOLO_IO_READER_INSTANTIATE_FIELD(float)
RETICOLO_IO_READER_INSTANTIATE_FIELD(double)
RETICOLO_IO_READER_INSTANTIATE_FIELD(std::complex<float>)
RETICOLO_IO_READER_INSTANTIATE_FIELD(std::complex<double>)
#undef RETICOLO_IO_READER_INSTANTIATE_FIELD

}  // namespace reticolo::io
