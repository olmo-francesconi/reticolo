#include <reticolo/io/writer.hpp>

#include <array>
#include <filesystem>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <hdf5.h>
#include <unistd.h>

namespace {

struct TempH5 {
    std::filesystem::path path;
    explicit TempH5(std::string const& tag) {
        path = std::filesystem::temp_directory_path() /
               ("reticolo_" + tag + "_" + std::to_string(::getpid()) + ".h5");
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    ~TempH5() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
    TempH5(TempH5 const&)            = delete;
    TempH5& operator=(TempH5 const&) = delete;
};

std::string read_string_attr(hid_t obj, char const* name) {
    hid_t a = H5Aopen(obj, name, H5P_DEFAULT);
    REQUIRE(a >= 0);
    hid_t s_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(s_type, H5T_VARIABLE);
    H5Tset_strpad(s_type, H5T_STR_NULLTERM);
    char* cstr = nullptr;
    REQUIRE(H5Aread(a, s_type, &cstr) >= 0);
    std::string out = cstr;
    H5free_memory(cstr);
    H5Tclose(s_type);
    H5Aclose(a);
    return out;
}

}  // namespace

using reticolo::io::Writer;

TEST_CASE("Writer stamps /run with all required attributes", "[io][metadata]") {
    TempH5 f{"metadata"};

    std::array<char const*, 3> argv{"phi4_hmc", "--L=8", "--kappa=0.18"};
    { Writer w{f.path, static_cast<int>(argv.size()), argv.data()}; }

    hid_t file = H5Fopen(f.path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    hid_t run = H5Gopen2(file, "/run", H5P_DEFAULT);
    REQUIRE(run >= 0);

    auto const cmdline       = read_string_attr(run, "cmdline");
    auto const version       = read_string_attr(run, "version");
    auto const commit        = read_string_attr(run, "commit");
    auto const build_type    = read_string_attr(run, "build_type");
    auto const compile_flags = read_string_attr(run, "compile_flags");
    auto const hostname      = read_string_attr(run, "hostname");
    auto const started_utc   = read_string_attr(run, "started_utc");
    auto const hdf5_lib      = read_string_attr(run, "hdf5_library_version");
    auto const complex_sch   = read_string_attr(run, "hdf5_complex_schema");

    H5Gclose(run);
    H5Fclose(file);

    REQUIRE(cmdline == "phi4_hmc --L=8 --kappa=0.18");
    REQUIRE_FALSE(version.empty());
    REQUIRE_FALSE(commit.empty());
    REQUIRE_FALSE(build_type.empty());
    REQUIRE_FALSE(hostname.empty());
    // ISO-8601 UTC: 20 chars, ends with 'Z'.
    REQUIRE(started_utc.size() == 20);
    REQUIRE(started_utc.back() == 'Z');
    // libhdf5 version: "<major>.<minor>.<patch>".
    REQUIRE(std::count(hdf5_lib.begin(), hdf5_lib.end(), '.') == 2);

    // The schema-mode stamp is the bug class the v3 plan calls out: if this
    // field disappears, compile-time flags that change on-disk layout become
    // silent and unreproducible.
    REQUIRE(complex_sch == "legacy_compound");

    // `compile_flags` may be empty in some build configs, but we still want
    // the attribute to exist (verified by read_string_attr above not throwing).
    (void)compile_flags;
}

TEST_CASE("Writer leaves empty cmdline when no argv is given", "[io][metadata]") {
    TempH5 f{"metadata_no_argv"};
    { Writer w{f.path}; }

    hid_t file = H5Fopen(f.path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    hid_t run  = H5Gopen2(file, "/run", H5P_DEFAULT);
    REQUIRE(read_string_attr(run, "cmdline").empty());
    H5Gclose(run);
    H5Fclose(file);
}
