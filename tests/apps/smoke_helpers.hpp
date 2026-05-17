// Shared helpers for the per-app HDF5 smoke tests. Header-only so the macro
// `BINARY` and Catch2 macros stay in the test TU.

#pragma once

#include <cstdlib>
#include <filesystem>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <hdf5.h>
#include <sys/wait.h>
#include <unistd.h>

namespace reticolo::test {

[[nodiscard]] inline std::filesystem::path scratch_path(std::string const& tag) {
    return std::filesystem::temp_directory_path() /
           ("reticolo_" + tag + "_" + std::to_string(::getpid()) + ".h5");
}

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
