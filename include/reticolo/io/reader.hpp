#pragma once

#include <reticolo/core/field/lattice.hpp>
#include <reticolo/core/field/matrix_link_lattice.hpp>
#include <reticolo/core/rng/fast_rng.hpp>
#include <reticolo/io/writer.hpp>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

// Read-only counterpart to io::Writer. Opens an HDF5 file produced by Writer
// (or any well-formed snapshot following the same schema) and refills lattice
// buffers and RNG state from it. Like Writer, hid_t never leaks out.

namespace reticolo::io {

class Reader {
public:
    explicit Reader(std::filesystem::path const& path);
    ~Reader();

    Reader(Reader const&)            = delete;
    Reader& operator=(Reader const&) = delete;
    Reader(Reader&&) noexcept;
    Reader& operator=(Reader&&) noexcept;

    [[nodiscard]] std::filesystem::path const& path() const noexcept;

    // True if the dataset / group / attribute at `path` exists. Attribute
    // paths use the same `group_or_dataset@name` form as Writer::attr.
    [[nodiscard]] bool has(std::string_view path) const;

    // Read an attribute. Same supported scalar set as Writer::attr.
    template <class T>
    [[nodiscard]] T attr(std::string_view path) const;

    // The lattice shape recorded with a previously-written field.
    [[nodiscard]] std::vector<std::size_t> field_shape(std::string_view path) const;

    // Refill `lat` from the dataset at `path`. Throws std::runtime_error on
    // shape, kind, or scalar-type mismatch. Element count must match
    // (caller is responsible for constructing the target lattice on the
    // shape returned by `field_shape`).
    template <class T>
    void field(std::string_view path, Lattice<T>& lat) const;

    template <class G, class T>
    void field(std::string_view path, MatrixLinkLattice<G, T>& lat) const {
        read_field_raw_(path,
                        lat.data(),
                        lat.ncomponents(),
                        scalar_kind_of<T>(),
                        Writer::FieldKind::matrix_link,
                        lat.shape(),
                        G::n_real_components);
    }

    // Restore a FastRng's full state (state words + cached normal).
    [[nodiscard]] FastRng rng_state(std::string_view path) const;

    // Read a multi-stream RNG state written by Writer::rng_streams. Validates
    // kind, stream count, and words-per-stream against the resuming StreamSet;
    // throws std::runtime_error on any mismatch (a resume must reconstruct the
    // set with the same family and n_streams). Returns the flat words,
    // driver stream first.
    [[nodiscard]] std::vector<std::uint64_t> rng_streams(std::string_view path,
                                                         std::string_view expected_kind,
                                                         std::size_t expected_n_streams,
                                                         std::size_t expected_n_words) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    void read_field_raw_(std::string_view path,
                         void* data_out,
                         std::size_t n_elems,
                         Writer::ScalarKind scalar_kind,
                         Writer::FieldKind kind,
                         std::vector<std::size_t> const& expected_shape,
                         std::size_t expected_n_components) const;
};

template <class T>
void Reader::field(std::string_view path, Lattice<T>& lat) const {
    read_field_raw_(path,
                    lat.data(),
                    lat.nsites(),
                    scalar_kind_of<T>(),
                    Writer::FieldKind::scalar,
                    lat.shape(),
                    1);
}

// Extern template declarations for the attribute reader. Same set as Writer.
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define RETICOLO_IO_READER_DECLARE(T) extern template T Reader::attr<T>(std::string_view) const;
RETICOLO_IO_READER_DECLARE(float)
RETICOLO_IO_READER_DECLARE(double)
RETICOLO_IO_READER_DECLARE(int)
RETICOLO_IO_READER_DECLARE(long)
RETICOLO_IO_READER_DECLARE(long long)
RETICOLO_IO_READER_DECLARE(unsigned int)
RETICOLO_IO_READER_DECLARE(unsigned long)
RETICOLO_IO_READER_DECLARE(unsigned long long)
#undef RETICOLO_IO_READER_DECLARE
extern template std::string Reader::attr<std::string>(std::string_view) const;

}  // namespace reticolo::io
