#include "../test_helpers.hpp"

#ifndef BOSE_GAS_LLR_BINARY
    #error "BOSE_GAS_LLR_BINARY compile definition is required"
#endif

using reticolo::test::require_link;
using reticolo::test::rows_in;
using reticolo::test::run_and_require_exit;
using reticolo::test::scratch_path;

// End-to-end: complex-LLR for the Bose gas — the window constrains the imaginary
// observable S_I (mode B). Same replica schema as u1_llr plus /cfg@mu.
TEST_CASE("bose_gas_llr binary writes the expected HDF5 schema", "[app][e2e][bose_gas_llr]") {
    auto const out = scratch_path("bose_gas_llr_smoke");
    std::error_code ec;
    std::filesystem::remove(out, ec);

    // Tiny config: symmetric S_I windows, 2 NR + 2 RM with short sample counts.
    std::string const cmd = std::string{BOSE_GAS_LLR_BINARY} +
                            " --size=4 --ndim=2 --mass=1.0 --lambda=1.0 --mu=0.5" +
                            " --E_min=-6 --E_max=6 --delta=3" + " --tau=1.0 --n_md=10" +
                            " --n_nr=2 --n_therm_nr=4 --n_meas_nr=8" +
                            " --n_rm=2 --n_therm_rm=4 --n_meas_rm=8" +
                            " --seed=20260701 --workspace=" + out.parent_path().string() +
                            " --out=" + out.filename().string();
    run_and_require_exit(cmd);
    REQUIRE(std::filesystem::exists(out));

    hid_t file = H5Fopen(out.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    require_link(file, "/run");
    require_link(file, "/cfg/E_n");
    require_link(file, "/replica_000/a");
    require_link(file, "/replica_000/dE");
    require_link(file, "/exchange/accepted");
    REQUIRE(rows_in(file, "/replica_000/a") == 4);  // n_nr + n_rm
    REQUIRE(rows_in(file, "/exchange/accepted") == 2);

    H5Fclose(file);
    std::filesystem::remove(out, ec);
}
