#include <cstdlib>
#include <filesystem>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <hdf5.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef PHI4_HMC_BINARY
    #error "PHI4_HMC_BINARY compile definition is required (path to the phi4_hmc binary)"
#endif

namespace {

std::filesystem::path scratch_path(std::string const& tag) {
    return std::filesystem::temp_directory_path() /
           ("reticolo_" + tag + "_" + std::to_string(::getpid()) + ".h5");
}

}  // namespace

TEST_CASE("phi4_hmc binary writes the expected HDF5 schema", "[app][e2e][phi4_hmc]") {
    auto const out = scratch_path("phi4_smoke");
    std::error_code ec;
    std::filesystem::remove(out, ec);

    constexpr int k_n_therm = 10;
    constexpr int k_n_prod  = 25;

    std::string const cmd = std::string{PHI4_HMC_BINARY} + " --size=4 --kappa=0.13 --lambda=0.02" +
                            " --n_therm=" + std::to_string(k_n_therm) +
                            " --n_prod=" + std::to_string(k_n_prod) +
                            " --seed=20260517 --out=" + out.string();

    int const raw_rc = std::system(cmd.c_str());  // NOLINT(concurrency-mt-unsafe)
    REQUIRE(WIFEXITED(raw_rc));
    REQUIRE(WEXITSTATUS(raw_rc) == 0);
    REQUIRE(std::filesystem::exists(out));

    hid_t file = H5Fopen(out.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);

    SECTION("reproducibility metadata is present") {
        REQUIRE(H5Lexists(file, "/run", H5P_DEFAULT) > 0);
        REQUIRE(H5Lexists(file, "/vars", H5P_DEFAULT) > 0);
    }
    SECTION("therm + prod phases exist") {
        REQUIRE(H5Lexists(file, "/therm", H5P_DEFAULT) > 0);
        REQUIRE(H5Lexists(file, "/prod", H5P_DEFAULT) > 0);
    }
    SECTION("every series is written with the right number of rows") {
        auto rows_in = [&](char const* path) -> hsize_t {
            hid_t dset = H5Dopen2(file, path, H5P_DEFAULT);
            REQUIRE(dset >= 0);
            hid_t space = H5Dget_space(dset);
            hsize_t n   = 0;
            H5Sget_simple_extent_dims(space, &n, nullptr);
            H5Sclose(space);
            H5Dclose(dset);
            return n;
        };
        REQUIRE(rows_in("/therm/stats/s") == static_cast<hsize_t>(k_n_therm));
        REQUIRE(rows_in("/prod/stats/dH") == static_cast<hsize_t>(k_n_prod));
        REQUIRE(rows_in("/prod/stats/accepted") == static_cast<hsize_t>(k_n_prod));
        REQUIRE(rows_in("/prod/obs/s") == static_cast<hsize_t>(k_n_prod));
        REQUIRE(rows_in("/prod/obs/mag") == static_cast<hsize_t>(k_n_prod));
        REQUIRE(rows_in("/prod/obs/m2") == static_cast<hsize_t>(k_n_prod));
    }

    H5Fclose(file);
    std::filesystem::remove(out, ec);
}
