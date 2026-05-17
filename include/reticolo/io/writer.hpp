#pragma once

#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

// =============================================================================
//  Public-facing HDF5 writer.
//
//  Design:
//   - <hdf5.h> never enters a TU that #includes this header. The full HDF5
//     state lives behind PIMPL in src/io/writer.cpp.
//   - `Writer` opens with truncate-or-fail and stamps `/run@*` metadata so
//     every output file is self-describing: cmdline, version, commit,
//     compile_flags, hostname, started_utc, hdf5_complex_schema,
//     hdf5_library_version. A schema-relevant compile flag must always end
//     up in `/run@` — that's the "silent data divergence" class the v3
//     plan calls out.
//   - `Series<T>` is the only way to write a time series. It owns a buffer
//     of `chunk` rows and flushes on full / dtor. Returned by value (not by
//     reference into a map) so writer state mutations can't invalidate it.
//   - `Series<T>` and the typed `Writer::series<T>` / `Writer::attr<T>` are
//     extern-templated for the supported scalar set below. Add a type by
//     editing both this declaration block AND the matching explicit
//     instantiation in writer.cpp — the linker will tell you if you forget.
// =============================================================================

namespace reticolo::cli {
class Parser;  // forward declaration; Writer accepts a Parser ptr for /vars@*
}

namespace reticolo::io {

class Writer;

template <class T>
class Series {
public:
    using value_type = T;

    Series() = default;
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
    template <class T>
    void attr(std::string_view path, T const& value);

    // Register a phase prefix. Throws std::runtime_error if `phase` already
    // exists (catches typo'd phase reuse early instead of silently overwriting).
    void start_phase(std::string_view phase);

    [[nodiscard]] std::filesystem::path const& path() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// -----------------------------------------------------------------------------
// Extern template declarations for the supported scalar set.
// -----------------------------------------------------------------------------
#define RETICOLO_IO_DECLARE(T)                                                                     \
    extern template class Series<T>;                                                               \
    extern template Series<T> Writer::series<T>(std::string_view, std::size_t);                    \
    extern template void Writer::attr<T>(std::string_view, T const&);

RETICOLO_IO_DECLARE(float)
RETICOLO_IO_DECLARE(double)
RETICOLO_IO_DECLARE(int)
RETICOLO_IO_DECLARE(long)
RETICOLO_IO_DECLARE(long long)
RETICOLO_IO_DECLARE(unsigned int)
RETICOLO_IO_DECLARE(unsigned long)
RETICOLO_IO_DECLARE(unsigned long long)
RETICOLO_IO_DECLARE(std::complex<float>)
RETICOLO_IO_DECLARE(std::complex<double>)

#undef RETICOLO_IO_DECLARE

// String attributes are a separate codepath (variable-length on disk).
extern template void Writer::attr<std::string>(std::string_view, std::string const&);

}  // namespace reticolo::io
