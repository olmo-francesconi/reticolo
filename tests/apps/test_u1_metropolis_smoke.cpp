#include "../test_helpers.hpp"

#ifndef U1_METROPOLIS_BINARY
    #error "U1_METROPOLIS_BINARY compile definition is required"
#endif

using reticolo::test::require_link;
using reticolo::test::rows_in;
using reticolo::test::run_and_require_exit;
using reticolo::test::scratch_path;

TEST_CASE("u1_metropolis binary writes the expected HDF5 schema", "[app][e2e][u1_metropolis]") {
    auto const out = scratch_path("u1_metropolis_smoke");
    std::error_code ec;
    std::filesystem::remove(out, ec);

    constexpr int k_n_therm = 5;
    constexpr int k_n_prod  = 15;

    std::string const cmd =
        std::string{U1_METROPOLIS_BINARY} + " --size=4 --ndim=2 --beta=1.0 --sigma=1.0" +
        " --n_therm=" + std::to_string(k_n_therm) + " --n_prod=" + std::to_string(k_n_prod) +
        " --seed=20260518 --workspace=" + out.parent_path().string() +
        " --out=" + out.filename().string();
    run_and_require_exit(cmd);
    REQUIRE(std::filesystem::exists(out));

    hid_t file = H5Fopen(out.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    require_link(file, "/run");
    require_link(file, "/vars");
    require_link(file, "/therm");
    require_link(file, "/prod");
    REQUIRE(rows_in(file, "/therm/stats/s") == static_cast<hsize_t>(k_n_therm));
    REQUIRE(rows_in(file, "/prod/stats/accept") == static_cast<hsize_t>(k_n_prod));
    REQUIRE(rows_in(file, "/prod/obs/s") == static_cast<hsize_t>(k_n_prod));
    REQUIRE(rows_in(file, "/prod/obs/plaq") == static_cast<hsize_t>(k_n_prod));

    H5Fclose(file);
    std::filesystem::remove(out, ec);
}
