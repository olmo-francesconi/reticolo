#include "../test_helpers.hpp"

#ifndef BOSE_GAS_HMC_CUDA_BINARY
    #error "BOSE_GAS_HMC_CUDA_BINARY compile definition is required"
#endif

using reticolo::test::require_link;
using reticolo::test::rows_in;
using reticolo::test::run_and_require_exit;
using reticolo::test::scratch_path;

// End-to-end: the CUDA phase-quenched Bose-gas HMC writes the same S_R + S_I
// schema as the host app (measurement per host-free block → rows = n/meas_every).
// Requires a GPU at run time. Mirrors test_bose_gas_hmc_smoke.cpp.
TEST_CASE("bose_gas_hmc_cuda binary writes S_R + S_I schema", "[app][e2e][cuda][bose_gas_cuda]") {
    auto const out = scratch_path("bose_gas_hmc_cuda_smoke");
    std::error_code ec;
    std::filesystem::remove(out, ec);

    constexpr int k_n_therm    = 10;
    constexpr int k_n_prod     = 20;
    constexpr int k_meas_every = 5;
    constexpr int k_therm_rows = k_n_therm / k_meas_every;  // 2
    constexpr int k_prod_rows  = k_n_prod / k_meas_every;   // 4

    std::string const cmd = std::string{BOSE_GAS_HMC_CUDA_BINARY} +
                            " --size=4 --ndim=2 --mass=1.0 --lambda=1.0 --mu=0.5" +
                            " --n_therm=" + std::to_string(k_n_therm) +
                            " --n_prod=" + std::to_string(k_n_prod) +
                            " --meas_every=" + std::to_string(k_meas_every) +
                            " --seed=20260701 --workspace=" + out.parent_path().string() +
                            " --out=" + out.filename().string();
    run_and_require_exit(cmd);
    REQUIRE(std::filesystem::exists(out));

    hid_t file = H5Fopen(out.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);

    require_link(file, "/run");
    require_link(file, "/prod/obs/s_r");
    require_link(file, "/prod/obs/s_i");

    REQUIRE(rows_in(file, "/therm/stats/s") == static_cast<hsize_t>(k_therm_rows));
    REQUIRE(rows_in(file, "/prod/obs/s_r") == static_cast<hsize_t>(k_prod_rows));
    REQUIRE(rows_in(file, "/prod/obs/s_i") == static_cast<hsize_t>(k_prod_rows));

    H5Fclose(file);
    std::filesystem::remove(out, ec);
}
