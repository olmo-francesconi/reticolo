#include "../test_helpers.hpp"

#ifndef PHI4_LLR_CUDA_BINARY
    #error "PHI4_LLR_CUDA_BINARY compile definition is required"
#endif

using reticolo::test::require_link;
using reticolo::test::rows_in;
using reticolo::test::run_and_require_exit;
using reticolo::test::scratch_path;

// End-to-end: the CUDA phi4 LLR app writes the LLR schema (per-slot a / dE / E_n
// series + exchange counts). Requires a GPU at run time. E_n is a per-slot
// series here (param-swap exchange migrates windows across slots). Mirrors
// test_u1_llr_smoke.cpp.
TEST_CASE("phi4_llr_cuda binary writes the expected HDF5 schema", "[app][e2e][cuda][phi4_llr]") {
    auto const out = scratch_path("phi4_llr_cuda_smoke");
    std::error_code ec;
    std::filesystem::remove(out, ec);

    // Tiny config: 4 replicas, 2 NR, 2 RM with very short sample counts.
    std::string const cmd =
        std::string{PHI4_LLR_CUDA_BINARY} + " --size=4 --ndim=2 --kappa=0.13 --lambda=0.02" +
        " --E_min=0 --E_max=30 --delta=10" + " --tau=1.0 --n_md=10" +
        " --n_nr=2 --n_therm_nr=4 --n_meas_nr=8" + " --n_rm=2 --n_therm_rm=4 --n_meas_rm=8" +
        " --seed=20260518 --workspace=" + out.parent_path().string() +
        " --out=" + out.filename().string();
    run_and_require_exit(cmd);
    REQUIRE(std::filesystem::exists(out));

    hid_t file = H5Fopen(out.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    require_link(file, "/run");
    require_link(file, "/vars");
    require_link(file, "/cfg/E_n");
    require_link(file, "/replica_000/a");
    require_link(file, "/replica_000/dE");
    require_link(file, "/replica_000/E_n");
    require_link(file, "/exchange/accepted");
    // n_nr + n_rm rows per replica.
    REQUIRE(rows_in(file, "/replica_000/a") == 4);
    REQUIRE(rows_in(file, "/replica_000/E_n") == 4);
    REQUIRE(rows_in(file, "/exchange/accepted") == 2);

    H5Fclose(file);
    std::filesystem::remove(out, ec);
}
