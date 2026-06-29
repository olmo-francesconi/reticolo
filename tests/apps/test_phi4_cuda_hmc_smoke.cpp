#include "../test_helpers.hpp"

#ifndef PHI4_CUDA_HMC_BINARY
    #error "PHI4_CUDA_HMC_BINARY compile definition is required"
#endif

using reticolo::test::require_link;
using reticolo::test::rows_in;
using reticolo::test::run_and_require_exit;
using reticolo::test::scratch_path;

// End-to-end: the CUDA phi4 app writes the same HDF5 schema as the host app
// (measurement is per host-free block, so row counts are n/meas_every). Requires
// a GPU at run time. Mirrors test_phi4_hmc_smoke.cpp.
TEST_CASE("phi4_cuda_hmc binary writes the expected HDF5 schema", "[app][e2e][cuda][phi4_cuda]") {
    auto const out = scratch_path("phi4_cuda_smoke");
    std::error_code ec;
    std::filesystem::remove(out, ec);

    constexpr int k_n_therm    = 10;
    constexpr int k_n_prod     = 20;
    constexpr int k_meas_every = 5;
    constexpr int k_therm_rows = k_n_therm / k_meas_every;  // 2
    constexpr int k_prod_rows  = k_n_prod / k_meas_every;   // 4

    std::string const cmd =
        std::string{PHI4_CUDA_HMC_BINARY} + " --size=4 --kappa=0.13 --lambda=0.02" +
        " --n_therm=" + std::to_string(k_n_therm) + " --n_prod=" + std::to_string(k_n_prod) +
        " --meas_every=" + std::to_string(k_meas_every) +
        " --seed=20260517 --workspace=" + out.parent_path().string() +
        " --out=" + out.filename().string();
    run_and_require_exit(cmd);
    REQUIRE(std::filesystem::exists(out));

    hid_t file = H5Fopen(out.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);

    require_link(file, "/run");
    require_link(file, "/vars");
    require_link(file, "/therm");
    require_link(file, "/prod");

    REQUIRE(rows_in(file, "/therm/stats/s") == static_cast<hsize_t>(k_therm_rows));
    REQUIRE(rows_in(file, "/prod/obs/s") == static_cast<hsize_t>(k_prod_rows));
    REQUIRE(rows_in(file, "/prod/obs/mag") == static_cast<hsize_t>(k_prod_rows));
    REQUIRE(rows_in(file, "/prod/obs/mag_sq") == static_cast<hsize_t>(k_prod_rows));
    REQUIRE(rows_in(file, "/prod/obs/m2") == static_cast<hsize_t>(k_prod_rows));

    H5Fclose(file);
    std::filesystem::remove(out, ec);
}
