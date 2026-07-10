#pragma once

#include <reticolo/core/lattice.hpp>
#include <reticolo/core/matrix_link_lattice.hpp>
#include <reticolo/core/rng/fast_rng.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

// Public-facing HDF5 writer.
//
// Design:
//  - <hdf5.h> never enters a TU that #includes this header. The full HDF5
//    state lives behind PIMPL in src/io/writer.cpp.
//  - `Writer` opens with truncate-or-fail and stamps `/run@*` metadata so
//    every output file is self-describing: cmdline, version, commit,
//    compile_flags, hostname, started_utc, hdf5_complex_schema,
//    hdf5_library_version. A schema-relevant compile flag must always end
//    up in `/run@` — that's the "silent data divergence" class the v3
//    plan calls out.
//  - `Series<T>` is the only way to write a time series. It owns a buffer
//    of `chunk` rows and flushes on full / dtor. Returned by value (not by
//    reference into a map) so writer state mutations can't invalidate it.
//  - `Series<T>` and the typed `Writer::series<T>` / `Writer::attr<T>` are
//    extern-templated for the supported scalar set below. Add a type by
//    editing both this declaration block AND the matching explicit
//    instantiation in writer.cpp — the linker will tell you if you forget.

namespace reticolo::cli {
class Parser;  // forward declaration; Writer accepts a Parser ptr for /vars@*
}

namespace reticolo::io {

class Writer;

template <class T>
class Series {
public:
    using value_type = T;

    Series();
    ~Series();

    Series(Series const&)            = delete;
    Series& operator=(Series const&) = delete;
    Series(Series&&) noexcept;
    Series& operator=(Series&&) noexcept;

    // Append one row. Buffered: writes to disk when the buffer reaches
    // `chunk` rows or when the Series is destroyed.
    void append(T const& value);

    // Force an immediate write of the buffered rows.
    void flush();

    // Total rows appended since construction (buffered + flushed).
    [[nodiscard]] std::size_t size() const noexcept;

    // True if this Series is bound to a writer (default-constructed is false).
    [[nodiscard]] bool valid() const noexcept;

private:
    friend class Writer;
    struct Impl;
    std::unique_ptr<Impl> impl_;

    explicit Series(std::unique_ptr<Impl> impl) noexcept;
};

class Writer {
public:
    // Open `path` with truncate-or-fail and stamp /run@* metadata.
    // `argc`/`argv` are recorded as /run@cmdline if given. If `params` is
    // non-null, every resolved var is stamped to /vars@<name> after the file
    // is initialised.
    explicit Writer(std::filesystem::path const& path,
                    int argc                  = 0,
                    char const* const* argv   = nullptr,
                    cli::Parser const* params = nullptr);

    ~Writer();

    Writer(Writer const&)            = delete;
    Writer& operator=(Writer const&) = delete;
    Writer(Writer&&) noexcept;
    Writer& operator=(Writer&&) noexcept;

    // Create a time-series dataset at `path` (HDF5 group hierarchy from "/").
    // Intermediate groups are created on demand.
    template <class T>
    [[nodiscard]] Series<T> series(std::string_view path, std::size_t chunk = 4096);

    // Write a scalar attribute at `path` of the form "<group_or_dataset>@<name>".
    // Examples: "/vars@kappa", "/run@operator_note".
    //
    // If the parent path doesn't exist, it is auto-created as an HDF5 group.
    // That's convenient for `/vars@*` and `/run@*` (the metadata roots), but
    // means a typo'd parent silently lands on a phantom new group instead of
    // failing. Treat `attr` as a metadata sink for known-flat namespaces, not
    // as a general dataset-attribute API.
    template <class T>
    void attr(std::string_view path, T const& value);

    // Register a phase prefix. Throws std::runtime_error if `phase` already
    // exists (catches typo'd phase reuse early instead of silently overwriting).
    void start_phase(std::string_view phase);

    // Write the full field buffer as a 1-D dataset at `path`. Companion
    // attributes record the kind ("scalar"/"link"/"matrix_link"), the scalar
    // type, the lattice shape, and the per-site component count. Together
    // they let `io::Reader::field` validate the target lattice and refill it
    // bit-exact. Element-count check: scalar = nsites; link = ndim·nsites;
    // matrix_link = ndim·nc·nsites.
    template <class T>
    void field(std::string_view path, Lattice<T> const& lat);

    template <class G, class T>
    void field(std::string_view path, MatrixLinkLattice<G, T> const& lat);

    // Write the RNG's full state (state words + cached normal) under `path`.
    // The corresponding Reader::rng_state restores it bit-exact. This
    // single-generator layout is FastRng-only; multi-stream generators
    // (StreamSet) go through `rng_streams` below.
    void rng_state(std::string_view path, FastRng const& rng);

    // Write a multi-stream RNG state (StreamSet checkpoint): one flat uint64
    // dataset of (n_streams+1)·n_words words — driver stream first — with
    // @kind / @n_streams / @n_words attributes. Reader::rng_streams validates
    // all three against the resuming StreamSet and returns the words.
    void rng_streams(std::string_view path,
                     std::string_view kind,
                     std::vector<std::uint64_t> const& words,
                     std::size_t n_streams,
                     std::size_t n_words);

    [[nodiscard]] std::filesystem::path const& path() const noexcept;

    // Wire-level scalar type tag. Public because the templated `field<T>`
    // helpers above translate `T` to this tag and hand it to the writer's
    // private raw-write path.
    enum class ScalarKind : std::uint8_t {
        f32,
        f64,
        c32,
        c64,
    };

    enum class FieldKind : std::uint8_t {
        scalar,
        link,
        matrix_link,
    };

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    void write_field_raw_(std::string_view path,
                          void const* data,
                          std::size_t n_elems,
                          ScalarKind scalar_kind,
                          FieldKind kind,
                          std::vector<std::size_t> const& shape,
                          std::size_t n_components,
                          char const* group_name);
};

template <class T>
constexpr Writer::ScalarKind scalar_kind_of();

template <>
constexpr Writer::ScalarKind scalar_kind_of<float>() {
    return Writer::ScalarKind::f32;
}
template <>
constexpr Writer::ScalarKind scalar_kind_of<double>() {
    return Writer::ScalarKind::f64;
}
template <>
constexpr Writer::ScalarKind scalar_kind_of<std::complex<float>>() {
    return Writer::ScalarKind::c32;
}
template <>
constexpr Writer::ScalarKind scalar_kind_of<std::complex<double>>() {
    return Writer::ScalarKind::c64;
}

template <class T>
void Writer::field(std::string_view path, Lattice<T> const& lat) {
    write_field_raw_(
        path, lat.data(), lat.nsites(), scalar_kind_of<T>(), FieldKind::scalar, lat.shape(), 1, "");
}

template <class G, class T>
void Writer::field(std::string_view path, MatrixLinkLattice<G, T> const& lat) {
    write_field_raw_(path,
                     lat.data(),
                     lat.ncomponents(),
                     scalar_kind_of<T>(),
                     FieldKind::matrix_link,
                     lat.shape(),
                     G::n_real_components,
                     G::name.data());
}

// Extern template declarations for the supported scalar set. The macro is the
// only sane way to write three correlated extern-template declarations per
// type; a `constexpr` template helper can't issue declarations.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define RETICOLO_IO_WRITER_DECLARE(T)                                                              \
    extern template class Series<T>;                                                               \
    extern template Series<T> Writer::series<T>(std::string_view, std::size_t);                    \
    extern template void Writer::attr<T>(std::string_view, T const&);

RETICOLO_IO_WRITER_DECLARE(float)
RETICOLO_IO_WRITER_DECLARE(double)
RETICOLO_IO_WRITER_DECLARE(int)
RETICOLO_IO_WRITER_DECLARE(long)
RETICOLO_IO_WRITER_DECLARE(long long)
RETICOLO_IO_WRITER_DECLARE(unsigned int)
RETICOLO_IO_WRITER_DECLARE(unsigned long)
RETICOLO_IO_WRITER_DECLARE(unsigned long long)
RETICOLO_IO_WRITER_DECLARE(std::complex<float>)
RETICOLO_IO_WRITER_DECLARE(std::complex<double>)

#undef RETICOLO_IO_WRITER_DECLARE

// String attributes are a separate codepath (variable-length on disk).
extern template void Writer::attr<std::string>(std::string_view, std::string const&);

}  // namespace reticolo::io
