#include "../test_helpers.hpp"

#ifndef BOSE_GAS_HMC_BINARY
    #error "BOSE_GAS_HMC_BINARY compile definition is required"
#endif

using reticolo::test::require_link;
using reticolo::test::rows_in;
using reticolo::test::run_and_require_exit;
using reticolo::test::scratch_path;

// End-to-end: the phase-quenched Bose-gas HMC records BOTH S_R and the imaginary
// observable S_I (the mode-B LLR twin), so the schema carries /prod/obs/s_r and
// /prod/obs/s_i.
TEST_CASE("bose_gas_hmc binary writes S_R + S_I schema", "[app][e2e][bose_gas_hmc]") {
    auto const out = scratch_path("bose_gas_hmc_smoke");
    std::error_code ec;
    std::filesystem::remove(out, ec);

    constexpr int k_n_therm = 8;
    constexpr int k_n_prod  = 12;

    std::string const cmd = std::string{BOSE_GAS_HMC_BINARY} +
                            " --size=4 --ndim=2 --mass=1.0 --lambda=1.0 --mu=0.5" +
                            " --n_therm=" + std::to_string(k_n_therm) +
                            " --n_prod=" + std::to_string(k_n_prod) + " --meas_every=1" +
                            " --seed=20260701 --workspace=" + out.parent_path().string() +
                            " --out=" + out.filename().string();
    run_and_require_exit(cmd);
    REQUIRE(std::filesystem::exists(out));

    hid_t file = H5Fopen(out.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);

    require_link(file, "/run");
    require_link(file, "/vars");
    require_link(file, "/prod/obs/s_r");
    require_link(file, "/prod/obs/s_i");

    REQUIRE(rows_in(file, "/therm/stats/s") == static_cast<hsize_t>(k_n_therm));
    REQUIRE(rows_in(file, "/prod/obs/s_r") == static_cast<hsize_t>(k_n_prod));
    REQUIRE(rows_in(file, "/prod/obs/s_i") == static_cast<hsize_t>(k_n_prod));

    H5Fclose(file);
    std::filesystem::remove(out, ec);
}
