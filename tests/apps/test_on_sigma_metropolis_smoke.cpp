#include "smoke_helpers.hpp"

#ifndef ON_SIGMA_METROPOLIS_BINARY
    #error "ON_SIGMA_METROPOLIS_BINARY compile definition is required"
#endif

using reticolo::test::require_link;
using reticolo::test::rows_in;
using reticolo::test::run_and_require_exit;
using reticolo::test::scratch_path;

TEST_CASE("on_sigma_metropolis binary writes the expected HDF5 schema",
          "[app][e2e][on_sigma_metropolis]") {
    auto const out = scratch_path("on_smoke");
    std::error_code ec;
    std::filesystem::remove(out, ec);

    constexpr int k_n_therm = 10;
    constexpr int k_n_prod  = 25;

    std::string const cmd =
        std::string{ON_SIGMA_METROPOLIS_BINARY} + " --size=4 --beta=1.0 --ndim=3" +
        " --n_therm=" + std::to_string(k_n_therm) + " --n_prod=" + std::to_string(k_n_prod) +
        " --seed=20260517 --out=" + out.string();
    run_and_require_exit(cmd);
    REQUIRE(std::filesystem::exists(out));

    hid_t file = H5Fopen(out.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);

    require_link(file, "/run");
    require_link(file, "/vars");
    require_link(file, "/therm");
    require_link(file, "/prod");

    REQUIRE(rows_in(file, "/therm/stats/accept") == static_cast<hsize_t>(k_n_therm));
    REQUIRE(rows_in(file, "/prod/stats/accept") == static_cast<hsize_t>(k_n_prod));
    REQUIRE(rows_in(file, "/prod/obs/s") == static_cast<hsize_t>(k_n_prod));
    REQUIRE(rows_in(file, "/prod/obs/m2") == static_cast<hsize_t>(k_n_prod));

    H5Fclose(file);
    std::filesystem::remove(out, ec);
}
