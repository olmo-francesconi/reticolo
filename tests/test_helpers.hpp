#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>

#include <catch2/catch_test_macros.hpp>
#include <hdf5.h>
#include <sys/wait.h>
#include <unistd.h>

namespace reticolo::test {

[[nodiscard]] inline std::filesystem::path scratch_path(std::string const& tag) {
    return std::filesystem::temp_directory_path() /
           ("reticolo_" + tag + "_" + std::to_string(::getpid()) + ".h5");
}

// RAII scratch HDF5 file: unique per (tag, pid), pre-cleared on construction,
// removed on destruction. Implicit-converts to filesystem::path for ergonomic
// passing to Writer / H5Fopen.
class ScratchH5 {
public:
    explicit ScratchH5(std::string const& tag) : path_{scratch_path(tag)} {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    ~ScratchH5() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    ScratchH5(ScratchH5 const&)            = delete;
    ScratchH5& operator=(ScratchH5 const&) = delete;
    ScratchH5(ScratchH5&&)                 = delete;
    ScratchH5& operator=(ScratchH5&&)      = delete;

    [[nodiscard]] std::filesystem::path const& path() const noexcept { return path_; }
    operator std::filesystem::path const&() const noexcept {
        return path_;
    }  // NOLINT(google-explicit-constructor)

private:
    std::filesystem::path path_;
};

inline void run_and_require_exit(std::string const& cmd) {
    int const raw_rc = std::system(cmd.c_str());  // NOLINT(concurrency-mt-unsafe)
    REQUIRE(WIFEXITED(raw_rc));
    REQUIRE(WEXITSTATUS(raw_rc) == 0);
}

[[nodiscard]] inline hsize_t rows_in(hid_t file, char const* path) {
    hid_t dset = H5Dopen2(file, path, H5P_DEFAULT);
    REQUIRE(dset >= 0);
    hid_t space = H5Dget_space(dset);
    hsize_t n   = 0;
    H5Sget_simple_extent_dims(space, &n, nullptr);
    H5Sclose(space);
    H5Dclose(dset);
    return n;
}

inline void require_link(hid_t file, char const* path) {
    REQUIRE(H5Lexists(file, path, H5P_DEFAULT) > 0);
}

}  // namespace reticolo::test
