#include "../test_helpers.hpp"

#include <reticolo/cli/parser.hpp>
#include <reticolo/io/writer.hpp>

#include <array>
#include <string>

#include <catch2/catch_test_macros.hpp>
#include <hdf5.h>

namespace {

double read_double_attr(hid_t obj, char const* name) {
    hid_t a = H5Aopen(obj, name, H5P_DEFAULT);
    REQUIRE(a >= 0);
    double v = 0.0;
    REQUIRE(H5Aread(a, H5T_NATIVE_DOUBLE, &v) >= 0);
    H5Aclose(a);
    return v;
}

int read_int_attr(hid_t obj, char const* name) {
    hid_t a = H5Aopen(obj, name, H5P_DEFAULT);
    REQUIRE(a >= 0);
    int v = 0;
    REQUIRE(H5Aread(a, H5T_NATIVE_INT, &v) >= 0);
    H5Aclose(a);
    return v;
}

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

namespace cli = reticolo::cli;
using reticolo::io::Writer;

TEST_CASE("Writer ctor stamps every Parser var at /vars@<name>", "[io][cli][vars]") {
    reticolo::test::ScratchH5 f{"vars_stamping"};

    cli::Parser p{"phi4_hmc"};
    auto const& L     = p.req<int>("L,size", "linear size");
    auto const& kappa = p.req<double>("kappa", "hopping");
    auto const& seed  = p.opt<unsigned long>("seed", 42UL);
    auto const& note  = p.opt<std::string>("note", std::string{"default"});

    std::array<char const*, 4> argv{"phi4_hmc", "--size=12", "--kappa=0.137", "--seed=7"};
    p.parse(static_cast<int>(argv.size()), argv.data());

    REQUIRE(L == 12);
    REQUIRE(kappa == 0.137);
    REQUIRE(seed == 7UL);
    REQUIRE(note == "default");

    { Writer w{f.path(), static_cast<int>(argv.size()), argv.data(), &p}; }

    hid_t file = H5Fopen(f.path().string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    hid_t vars = H5Gopen2(file, "/vars", H5P_DEFAULT);
    REQUIRE(vars >= 0);

    REQUIRE(read_int_attr(vars, "size") == 12);
    REQUIRE(read_double_attr(vars, "kappa") == 0.137);
    REQUIRE(read_string_attr(vars, "note") == "default");
    // seed: unsigned long; HDF5 stored it as H5T_NATIVE_ULONG. Read back same.
    hid_t a = H5Aopen(vars, "seed", H5P_DEFAULT);
    REQUIRE(a >= 0);
    unsigned long seed_read = 0;
    REQUIRE(H5Aread(a, H5T_NATIVE_ULONG, &seed_read) >= 0);
    H5Aclose(a);
    REQUIRE(seed_read == 7UL);

    H5Gclose(vars);
    H5Fclose(file);
}

TEST_CASE("Writer ctor without a Parser does not create /vars", "[io][cli][vars]") {
    reticolo::test::ScratchH5 f{"vars_absent"};

    { Writer w{f.path()}; }

    hid_t file = H5Fopen(f.path().string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    REQUIRE(H5Lexists(file, "/vars", H5P_DEFAULT) <= 0);
    H5Fclose(file);
}
