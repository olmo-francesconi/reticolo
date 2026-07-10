#pragma once

// Shared driver for the LLR --resume regression tests. An uninterrupted run and
// a checkpoint+resume run must produce a bit-identical RM tail — every piece of
// state a resume restores (per-replica field + HMC StreamSet + a, plus the
// orchestrator exch_rng and its schedule position) is pinned by comparing
// /replica_NNN/a, /replica_NNN/dE and /exchange/accepted. If a future change
// forgets one, these diverge.
//
// Output is segmented (a fresh Writer per launch), so the resumed run's series
// hold only the post-resume rows: full's RM sweep `s` lives at /a index
// (n_nr + s), and seg2's row i is full's sweep (ckpt_sweep + i).

#include "../test_helpers.hpp"

#include <cstddef>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <hdf5.h>

namespace reticolo::test {

[[nodiscard]] inline std::vector<double> read_doubles(hid_t file, std::string const& path) {
    hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
    REQUIRE(dset >= 0);
    hid_t space = H5Dget_space(dset);
    hsize_t n   = 0;
    H5Sget_simple_extent_dims(space, &n, nullptr);
    std::vector<double> out(n);
    REQUIRE(H5Dread(dset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) >= 0);
    H5Sclose(space);
    H5Dclose(dset);
    return out;
}

[[nodiscard]] inline std::vector<int> read_ints(hid_t file, std::string const& path) {
    hid_t dset = H5Dopen2(file, path.c_str(), H5P_DEFAULT);
    REQUIRE(dset >= 0);
    hid_t space = H5Dget_space(dset);
    hsize_t n   = 0;
    H5Sget_simple_extent_dims(space, &n, nullptr);
    std::vector<int> out(n);
    REQUIRE(H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) >= 0);
    H5Sclose(space);
    H5Dclose(dset);
    return out;
}

// Run <binary phys_args> three ways — full (n_rm sweeps), seg1 (ckpt_sweep
// sweeps, rolling checkpoint), seg2 (resume from that checkpoint out to n_rm) —
// and require the RM tails bit-identical. `phys_args` carries lattice / window /
// therm / meas / seed / --replica_threads, but NOT n_nr, n_rm, workspace, out,
// checkpoint or resume (built here). `omp_threads` sets OMP_NUM_THREADS so an
// m>1 case actually nests on an OpenMP build (harmless serial otherwise).
inline void require_llr_resume_equivalence(std::string const& binary,
                                           std::string const& phys_args,
                                           int n_nr,
                                           int n_rm,
                                           int ckpt_sweep,
                                           std::string const& tag,
                                           int omp_threads = 1) {
    namespace fs          = std::filesystem;
    std::string const d   = fs::temp_directory_path().string();
    std::string const pid = std::to_string(::getpid());
    auto name = [&](char const* p) { return "reticolo_" + tag + "_" + p + "_" + pid + ".h5"; };
    std::string const full_out = name("full");
    std::string const seg2_out = name("seg2");
    std::string const ckpt     = d + "/" + name("ckpt");

    std::string const env  = "OMP_NUM_THREADS=" + std::to_string(omp_threads) + " ";
    std::string const base = env + binary + " --n_nr=" + std::to_string(n_nr) + " " + phys_args +
                             " --workspace=" + d + " ";

    run_and_require_exit(base + "--n_rm=" + std::to_string(n_rm) + " --out=" + full_out);
    run_and_require_exit(base + "--n_rm=" + std::to_string(ckpt_sweep) + " --out=" + name("seg1") +
                         " --checkpoint=" + ckpt + " --checkpoint_every=1");
    run_and_require_exit(base + "--n_rm=" + std::to_string(n_rm) + " --out=" + seg2_out +
                         " --resume=" + ckpt);

    hid_t full = H5Fopen((d + "/" + full_out).c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t seg2 = H5Fopen((d + "/" + seg2_out).c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(full >= 0);
    REQUIRE(seg2 >= 0);

    auto const n_rep   = static_cast<int>(rows_in(full, "/cfg/E_n"));
    int const tail_len = n_rm - ckpt_sweep;
    REQUIRE(tail_len > 0);

    auto require_tail = [&](std::string const& path, int full_offset) {
        auto const x = read_doubles(full, path);
        auto const y = read_doubles(seg2, path);
        REQUIRE(y.size() == static_cast<std::size_t>(tail_len));
        for (int i = 0; i < tail_len; ++i) {
            INFO(path << " mismatch at tail i=" << i);
            REQUIRE(x.at(static_cast<std::size_t>(full_offset + i)) ==
                    y.at(static_cast<std::size_t>(i)));
        }
    };

    for (int r = 0; r < n_rep; ++r) {
        char grp[24];
        std::snprintf(grp, sizeof(grp), "/replica_%03d", r);
        require_tail(std::string{grp} + "/a", n_nr + ckpt_sweep);
        require_tail(std::string{grp} + "/dE", n_nr + ckpt_sweep);
    }
    // Exchange series carries one row per RM sweep (no NR rows).
    auto const xf = read_ints(full, "/exchange/accepted");
    auto const xs = read_ints(seg2, "/exchange/accepted");
    REQUIRE(xs.size() == static_cast<std::size_t>(tail_len));
    for (int i = 0; i < tail_len; ++i) {
        INFO("/exchange/accepted mismatch at tail i=" << i);
        REQUIRE(xf.at(static_cast<std::size_t>(ckpt_sweep + i)) ==
                xs.at(static_cast<std::size_t>(i)));
    }

    H5Fclose(full);
    H5Fclose(seg2);

    std::error_code ec;
    for (auto const* n : {"full", "seg1", "seg2", "ckpt"}) {
        fs::remove(d + "/" + name(n), ec);
    }
}

}  // namespace reticolo::test
