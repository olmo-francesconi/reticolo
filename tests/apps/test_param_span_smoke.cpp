#include "../test_helpers.hpp"

#ifndef PARAM_SPAN_HMC_BINARY
    #error "PARAM_SPAN_HMC_BINARY compile definition is required"
#endif

using reticolo::test::require_link;
using reticolo::test::rows_in;
using reticolo::test::run_and_require_exit;
using reticolo::test::scratch_path;

// End-to-end: the param-span orchestrator drives N concurrent phi^4 HMC workers
// over a kappa span through the generic orch:: spine, writing one HDF5 file with
// a /worker_NNN/ group per kappa point.
TEST_CASE("param_span_hmc binary writes the expected HDF5 schema", "[app][e2e][param_span]") {
    auto const out = scratch_path("param_span_smoke");
    std::error_code ec;
    std::filesystem::remove(out, ec);

    constexpr int k_n_workers = 4;
    constexpr int k_n_therm   = 10;
    constexpr int k_n_prod    = 20;

    std::string const cmd =
        std::string{PARAM_SPAN_HMC_BINARY} + " --size=4 --ndim=2 --lambda=0.02" +
        " --kappa_min=0.10 --kappa_max=0.24 --n_workers=" + std::to_string(k_n_workers) +
        " --n_therm=" + std::to_string(k_n_therm) + " --n_prod=" + std::to_string(k_n_prod) +
        " --seed=20260711 --workspace=" + out.parent_path().string() +
        " --out=" + out.filename().string();
    run_and_require_exit(cmd);
    REQUIRE(std::filesystem::exists(out));

    hid_t file = H5Fopen(out.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);

    require_link(file, "/run");
    require_link(file, "/vars");
    require_link(file, "/cfg/kappa");
    REQUIRE(rows_in(file, "/cfg/kappa") == static_cast<hsize_t>(k_n_workers));

    // First and last worker groups exist and carry the full stats + obs schema.
    require_link(file, "/worker_000/stats/dH");
    require_link(file, "/worker_003/obs/s");
    REQUIRE(rows_in(file, "/worker_000/stats/dH") == static_cast<hsize_t>(k_n_prod));
    REQUIRE(rows_in(file, "/worker_000/stats/accepted") == static_cast<hsize_t>(k_n_prod));
    REQUIRE(rows_in(file, "/worker_003/obs/s") == static_cast<hsize_t>(k_n_prod));
    REQUIRE(rows_in(file, "/worker_003/obs/mag") == static_cast<hsize_t>(k_n_prod));
    REQUIRE(rows_in(file, "/worker_003/obs/m2") == static_cast<hsize_t>(k_n_prod));

    H5Fclose(file);
    std::filesystem::remove(out, ec);
}
