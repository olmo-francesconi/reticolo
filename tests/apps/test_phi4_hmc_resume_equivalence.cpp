// End-to-end regression test for io::save_config / --resume:
// a baseline run and a checkpoint+resume run must produce bit-identical
// production streams. The check pins every piece of state we have to
// snapshot — field, RNG (including the Box-Muller cached spare), and the
// trajectory counter. If a future change forgets one of those, this test
// will diverge.

#include "../test_helpers.hpp"

#include <complex>
#include <cstddef>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <hdf5.h>

#ifndef PHI4_HMC_BINARY
    #error "PHI4_HMC_BINARY compile definition is required"
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
    char buf[16];
    std::snprintf(buf, sizeof(buf), ".cfg.%05lld.h5", i);
    return stem + buf;
}

}  // namespace

TEST_CASE("phi4_hmc --resume reproduces the would-have-been continuation bit-exact",
          "[app][e2e][phi4_hmc][checkpoint][regression]") {
    auto const base_out  = scratch_path("phi4_resume_base");
    auto const part2_out = scratch_path("phi4_resume_part2");
    auto const cfg15     = cfg_path(base_out, 15);
    auto const cfg30     = cfg_path(base_out, 30);

    auto cleanup = [&]() {
        std::error_code ec;
        for (auto const& p : {base_out, part2_out, cfg15, cfg30}) {
            std::filesystem::remove(p, ec);
        }
    };
    cleanup();

    constexpr int k_n_therm      = 30;
    constexpr int k_n_prod       = 30;
    constexpr int k_ckpt         = 15;
    constexpr char const* k_seed = "20260518";

    std::string const base_cmd = std::string{PHI4_HMC_BINARY} +
                                 " -L 6 --ndim=3 --kappa=0.13 --lambda=0.02"
                                 " --n_therm=" +
                                 std::to_string(k_n_therm) +
                                 " --n_prod=" + std::to_string(k_n_prod) +
                                 " --checkpoint_every=" + std::to_string(k_ckpt) +
                                 " --seed=" + k_seed + " --out=" + base_out.string();
    run_and_require_exit(base_cmd);

    REQUIRE(std::filesystem::exists(cfg15));

    std::string const resume_cmd = std::string{PHI4_HMC_BINARY} +
                                   " -L 6 --ndim=3 --kappa=0.13 --lambda=0.02"
                                   " --n_prod=" +
                                   std::to_string(k_n_prod) + " --resume=" + cfg15.string() +
                                   " --seed=" + k_seed + " --out=" + part2_out.string();
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
    check_doubles("/prod/obs/mag");
    check_doubles("/prod/obs/mag_sq");
    check_doubles("/prod/obs/m2");

    H5Fclose(a);
    H5Fclose(b);
    cleanup();
}
