// End-to-end regression test for io::save_config / --resume on a GAUGE HMC app.
//
// The scalar twin (test_phi4_hmc_resume_equivalence) covers the Lattice<double>
// path; gauge HMC resume had no cover at all until this test — only gauge LLR
// resume (test_su2_llr_resume_equivalence) did. It arrived with the app-layer
// sweep that gave all 16 HMC apps `app::resume_or_start`.
//
// What this actually pins is the MatrixLinkLattice round-trip: the SoA
// per-component link storage must save and reload exactly (the component stride
// is `link_span()`, which may exceed nsites under the opt-in padding), alongside
// the RNG state (including the Box-Muller cached spare) and the trajectory
// counter. A baseline run and a checkpoint+resume run must then produce
// bit-identical production streams.
//
// NOTE it does NOT pin the apps' `if (!resuming)` cold-start guard: that runs
// before `resume_or_start`, which overwrites the links anyway, so the guard
// skips wasted work rather than preventing corruption. Verified by removing it
// and watching this test still pass.

#include "../test_helpers.hpp"

#include <array>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <hdf5.h>

#ifndef SU2_HMC_BINARY
    #error "SU2_HMC_BINARY compile definition is required"
#endif

using reticolo::test::run_and_require_exit;
using reticolo::test::scratch_path;

namespace {

std::vector<double> read_doubles(hid_t file, char const* path) {
    hid_t dset = H5Dopen2(file, path, H5P_DEFAULT);
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

std::vector<int> read_ints(hid_t file, char const* path) {
    hid_t dset = H5Dopen2(file, path, H5P_DEFAULT);
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

std::filesystem::path cfg_path(std::filesystem::path const& base, long long i) {
    std::string stem = base.string();
    if (auto const pos = stem.rfind(".h5"); pos != std::string::npos && pos == stem.size() - 3) {
        stem.resize(pos);
    }
    std::array<char, 16> buf{};
    std::snprintf(buf.data(), buf.size(), ".cfg.%05lld.h5", i);
    return stem + buf.data();
}

}  // namespace

TEST_CASE("su2_hmc --resume reproduces the would-have-been continuation bit-exact",
          "[app][e2e][su2_hmc][checkpoint][regression]") {
    auto const base_out  = scratch_path("su2_resume_base");
    auto const part2_out = scratch_path("su2_resume_part2");
    auto const cfg10     = cfg_path(base_out, 10);
    auto const cfg20     = cfg_path(base_out, 20);

    auto cleanup = [&]() {
        std::error_code ec;
        for (auto const& p : {base_out, part2_out, cfg10, cfg20}) {
            std::filesystem::remove(p, ec);
        }
    };
    cleanup();

    constexpr int k_n_therm      = 10;
    constexpr int k_n_prod       = 20;
    constexpr int k_ckpt         = 10;
    constexpr char const* k_seed = "20260801";

    // ndim=3, L=4 keeps the gauge trajectory cheap; n_md is small for the same
    // reason. The point is the resume path, not the physics.
    std::string const common = " -L 4 --ndim=3 --beta=2.0 --n_md=4 --seed=" + std::string{k_seed};

    std::string const base_cmd =
        std::string{SU2_HMC_BINARY} + common + " --n_therm=" + std::to_string(k_n_therm) +
        " --n_prod=" + std::to_string(k_n_prod) + " --checkpoint_every=" + std::to_string(k_ckpt) +
        " --workspace=" + base_out.parent_path().string() +
        " --out=" + base_out.filename().string();
    run_and_require_exit(base_cmd);

    REQUIRE(std::filesystem::exists(cfg10));

    // No --n_therm: a resumed run skips thermalisation and picks the loaded
    // links up exactly where the checkpoint left them.
    std::string const resume_cmd =
        std::string{SU2_HMC_BINARY} + common + " --n_prod=" + std::to_string(k_n_prod) +
        " --resume=" + cfg10.string() + " --workspace=" + part2_out.parent_path().string() +
        " --out=" + part2_out.filename().string();
    run_and_require_exit(resume_cmd);

    hid_t a = H5Fopen(base_out.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t b = H5Fopen(part2_out.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(a >= 0);
    REQUIRE(b >= 0);

    auto check_doubles = [&](char const* path) {
        auto const x = read_doubles(a, path);
        auto const y = read_doubles(b, path);
        REQUIRE(x.size() == static_cast<std::size_t>(k_n_prod));
        REQUIRE(y.size() == static_cast<std::size_t>(k_n_prod - k_ckpt));
        for (std::size_t i = 0; i < y.size(); ++i) {
            INFO(path << " mismatch at i=" << i);
            REQUIRE(x[k_ckpt + i] == y[i]);
        }
    };
    auto check_ints = [&](char const* path) {
        auto const x = read_ints(a, path);
        auto const y = read_ints(b, path);
        REQUIRE(x.size() == static_cast<std::size_t>(k_n_prod));
        REQUIRE(y.size() == static_cast<std::size_t>(k_n_prod - k_ckpt));
        for (std::size_t i = 0; i < y.size(); ++i) {
            INFO(path << " mismatch at i=" << i);
            REQUIRE(x[k_ckpt + i] == y[i]);
        }
    };

    check_doubles("/prod/stats/dH");
    check_ints("/prod/stats/accepted");
    check_doubles("/prod/obs/s");
    check_doubles("/prod/obs/plaq");

    H5Fclose(a);
    H5Fclose(b);
    cleanup();
}
