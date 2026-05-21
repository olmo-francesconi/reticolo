#include "../test_helpers.hpp"

#ifndef U1_HMC_BINARY
    #error "U1_HMC_BINARY compile definition is required"
#endif

using reticolo::test::require_link;
using reticolo::test::rows_in;
using reticolo::test::run_and_require_exit;
using reticolo::test::scratch_path;

TEST_CASE("u1_hmc binary writes the expected HDF5 schema", "[app][e2e][u1_hmc]") {
    auto const out = scratch_path("u1_hmc_smoke");
    std::error_code ec;
    std::filesystem::remove(out, ec);

    constexpr int k_n_therm = 5;
    constexpr int k_n_prod  = 15;

    std::string const cmd =
        std::string{U1_HMC_BINARY} + " --size=4 --ndim=2 --beta=1.0 --tau=1.0 --n_md=10" +
        " --n_therm=" + std::to_string(k_n_therm) + " --n_prod=" + std::to_string(k_n_prod) +
        " --seed=20260518 --out=" + out.string();
    run_and_require_exit(cmd);
    REQUIRE(std::filesystem::exists(out));

    hid_t file = H5Fopen(out.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    require_link(file, "/run");
    require_link(file, "/vars");
    require_link(file, "/therm");
    require_link(file, "/prod");
    REQUIRE(rows_in(file, "/therm/stats/s") == static_cast<hsize_t>(k_n_therm));
    REQUIRE(rows_in(file, "/prod/stats/dH") == static_cast<hsize_t>(k_n_prod));
    REQUIRE(rows_in(file, "/prod/stats/accepted") == static_cast<hsize_t>(k_n_prod));
    REQUIRE(rows_in(file, "/prod/obs/s") == static_cast<hsize_t>(k_n_prod));
    REQUIRE(rows_in(file, "/prod/obs/plaq") == static_cast<hsize_t>(k_n_prod));

    H5Fclose(file);
    std::filesystem::remove(out, ec);
}
