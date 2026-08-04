#pragma once

// Shared end-to-end smoke drivers for the reference apps. Every smoke test runs
// the real binary on a tiny lattice and inspects the HDF5 it wrote — the check
// that the app's series wiring still matches the documented schema. That check
// is IDENTICAL across a family; only the binary, the couplings, and the
// per-action observable names differ. Single-sourced here so a schema change is
// one edit rather than seven, and so a new app's smoke test is a table entry.
//
// Two families, mirroring the two `app::*_run_flags` helpers:
//   * `require_hmc_smoke`  — /therm + /prod, stats + obs row counts
//   * `require_llr_smoke`  — /cfg/E_n + /replica_NNN + /exchange row counts

#include "../test_helpers.hpp"

#include <string>
#include <vector>

namespace reticolo::test {

struct HmcSmoke {
    std::string binary;
    std::string tag;
    // Couplings and any app-specific flags, e.g. "--kappa=0.13 --lambda=0.02".
    std::string physics;
    // Observable dataset names under /prod/obs. Every HMC app writes at least
    // one; the names are what differ per action (mag/m2, plaq, cos_phi, s_r/s_i).
    std::vector<std::string> obs;
    int n_therm             = 10;
    int n_prod              = 25;
    unsigned long long seed = 20260517;
    int size                = 4;
    // Some apps write /therm/stats/s, the complex ones use a different key.
    std::string therm_stat = "s";
};

inline void require_hmc_smoke(HmcSmoke const& s) {
    ScratchH5 const out{s.tag};

    std::string const cmd =
        s.binary + " --size=" + std::to_string(s.size) + " " + s.physics +
        " --n_therm=" + std::to_string(s.n_therm) + " --n_prod=" + std::to_string(s.n_prod) +
        " --seed=" + std::to_string(s.seed) + " --workspace=" + out.path().parent_path().string() +
        " --out=" + out.path().filename().string();
    run_and_require_exit(cmd);
    REQUIRE(std::filesystem::exists(out.path()));

    hid_t file = H5Fopen(out.path().string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);

    require_link(file, "/run");
    require_link(file, "/vars");
    require_link(file, "/therm");
    require_link(file, "/prod");

    auto const therm = static_cast<hsize_t>(s.n_therm);
    auto const prod  = static_cast<hsize_t>(s.n_prod);
    REQUIRE(rows_in(file, ("/therm/stats/" + s.therm_stat).c_str()) == therm);
    REQUIRE(rows_in(file, "/prod/stats/dH") == prod);
    REQUIRE(rows_in(file, "/prod/stats/accepted") == prod);
    for (std::string const& o : s.obs) {
        INFO("observable /prod/obs/" << o);
        require_link(file, ("/prod/obs/" + o).c_str());
        REQUIRE(rows_in(file, ("/prod/obs/" + o).c_str()) == prod);
    }

    H5Fclose(file);
}

struct LlrSmoke {
    std::string binary;
    std::string tag;
    // Couplings, e.g. "--beta=1.0" or "--mass=1.0 --lambda=1.0 --mu=0.5".
    std::string physics;
    // Window ladder, e.g. "--E_min=0 --E_max=30 --delta=10".
    std::string window;
    int ndim                = 2;
    int size                = 4;
    int n_nr                = 2;
    int n_rm                = 2;
    unsigned long long seed = 20260518;
    // Kept tiny: the seating phase is not what a schema smoke test is checking.
    int warm_therm    = 4;
    int warm_max_traj = 40;
};

inline void require_llr_smoke(LlrSmoke const& s) {
    ScratchH5 const out{s.tag};

    std::string const cmd =
        s.binary + " --size=" + std::to_string(s.size) + " --ndim=" + std::to_string(s.ndim) + " " +
        s.physics + " " + s.window + " --tau=1.0 --n_md=10" + " --n_nr=" + std::to_string(s.n_nr) +
        " --n_therm_nr=4 --n_meas_nr=8" + " --n_rm=" + std::to_string(s.n_rm) +
        " --n_therm_rm=4 --n_meas_rm=8" + " --warm_therm=" + std::to_string(s.warm_therm) +
        " --warm_max_traj=" + std::to_string(s.warm_max_traj) +
        " --seed=" + std::to_string(s.seed) + " --workspace=" + out.path().parent_path().string() +
        " --out=" + out.path().filename().string();
    run_and_require_exit(cmd);
    REQUIRE(std::filesystem::exists(out.path()));

    hid_t file = H5Fopen(out.path().string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);

    require_link(file, "/run");
    require_link(file, "/vars");
    require_link(file, "/cfg/E_n");
    require_link(file, "/replica_000/a");
    require_link(file, "/replica_000/dE");
    require_link(file, "/exchange/accepted");
    // One a/dE row per NR iteration and per RM sweep; exchange only per RM sweep.
    REQUIRE(rows_in(file, "/replica_000/a") == static_cast<hsize_t>(s.n_nr + s.n_rm));
    REQUIRE(rows_in(file, "/exchange/accepted") == static_cast<hsize_t>(s.n_rm));

    H5Fclose(file);
}

}  // namespace reticolo::test
