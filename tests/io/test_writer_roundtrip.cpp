#include <reticolo/io/writer.hpp>

#include "../test_helpers.hpp"

#include <complex>
#include <cstddef>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <hdf5.h>

namespace {

// Read a 1D dataset of T from `file` at `path`. Asserts on any HDF5 failure.
template <class T>
std::vector<T> read_series(hid_t file, char const* path, hid_t native) {
    hid_t dset = H5Dopen2(file, path, H5P_DEFAULT);
    REQUIRE(dset >= 0);

    hid_t space = H5Dget_space(dset);
    REQUIRE(space >= 0);
    hsize_t n = 0;
    H5Sget_simple_extent_dims(space, &n, nullptr);

    std::vector<T> out(n);
    REQUIRE(H5Dread(dset, native, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) >= 0);

    H5Sclose(space);
    H5Dclose(dset);
    return out;
}

}  // namespace

using reticolo::io::Series;
using reticolo::io::Writer;

TEST_CASE("Writer round-trips a Series<double> through file close", "[io][roundtrip]") {
    reticolo::test::ScratchH5 f{"roundtrip_double"};

    constexpr std::size_t k_n = 5000;
    {
        Writer w{f.path()};
        auto s = w.series<double>("/prod/obs/s");
        for (std::size_t i = 0; i < k_n; ++i) {
            s.append(static_cast<double>(i) * 0.5);
        }
    }

    hid_t file = H5Fopen(f.path().string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    auto read = read_series<double>(file, "/prod/obs/s", H5T_NATIVE_DOUBLE);
    H5Fclose(file);

    REQUIRE(read.size() == k_n);
    for (std::size_t i = 0; i < k_n; ++i) {
        REQUIRE(read[i] == static_cast<double>(i) * 0.5);
    }
}

TEST_CASE("Writer round-trips a Series<int>", "[io][roundtrip]") {
    reticolo::test::ScratchH5 f{"roundtrip_int"};

    {
        Writer w{f.path()};
        auto s = w.series<int>("/therm/stats/accept");
        for (int i = 0; i < 100; ++i) {
            s.append(i * 3);
        }
    }

    hid_t file = H5Fopen(f.path().string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    auto read = read_series<int>(file, "/therm/stats/accept", H5T_NATIVE_INT);
    H5Fclose(file);

    REQUIRE(read.size() == 100);
    for (int i = 0; i < 100; ++i) {
        REQUIRE(read[static_cast<std::size_t>(i)] == i * 3);
    }
}

TEST_CASE("Writer round-trips a Series<complex<double>> via legacy compound", "[io][roundtrip]") {
    reticolo::test::ScratchH5 f{"roundtrip_complex"};

    constexpr std::size_t k_n = 50;
    {
        Writer w{f.path()};
        auto s = w.series<std::complex<double>>("/prod/obs/z");
        for (std::size_t i = 0; i < k_n; ++i) {
            s.append(std::complex<double>{static_cast<double>(i), -static_cast<double>(i) * 0.1});
        }
    }

    // Reopen and rebuild the matching compound type for read-back.
    hid_t file = H5Fopen(f.path().string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);

    hid_t cid = H5Tcreate(H5T_COMPOUND, sizeof(std::complex<double>));
    H5Tinsert(cid, "r", 0, H5T_NATIVE_DOUBLE);
    H5Tinsert(cid, "i", sizeof(double), H5T_NATIVE_DOUBLE);

    auto read = read_series<std::complex<double>>(file, "/prod/obs/z", cid);
    H5Tclose(cid);
    H5Fclose(file);

    REQUIRE(read.size() == k_n);
    for (std::size_t i = 0; i < k_n; ++i) {
        REQUIRE(read[i].real() == static_cast<double>(i));
        REQUIRE(read[i].imag() == -static_cast<double>(i) * 0.1);
    }
}

TEST_CASE("Series flushes buffered rows when destroyed before chunk fills",
          "[io][roundtrip][flush]") {
    reticolo::test::ScratchH5 f{"flush_partial"};

    // Chunk = 1024 rows; we append only 17. Dtor must flush them.
    {
        Writer w{f.path()};
        auto s = w.series<double>("/probe", /*chunk=*/1024);
        for (int i = 0; i < 17; ++i) {
            s.append(static_cast<double>(i));
        }
    }

    hid_t file = H5Fopen(f.path().string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    auto read = read_series<double>(file, "/probe", H5T_NATIVE_DOUBLE);
    H5Fclose(file);

    REQUIRE(read.size() == 17);
    for (int i = 0; i < 17; ++i) {
        REQUIRE(read[static_cast<std::size_t>(i)] == static_cast<double>(i));
    }
}

TEST_CASE("Multiple Series in one Writer write independent datasets", "[io][roundtrip]") {
    reticolo::test::ScratchH5 f{"multi_series"};

    {
        Writer w{f.path()};
        auto a = w.series<double>("/a");
        auto b = w.series<int>("/b");
        for (int i = 0; i < 10; ++i) {
            a.append(static_cast<double>(i) + 0.25);
            b.append(i * 7);
        }
    }

    hid_t file = H5Fopen(f.path().string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    auto a_read = read_series<double>(file, "/a", H5T_NATIVE_DOUBLE);
    auto b_read = read_series<int>(file, "/b", H5T_NATIVE_INT);
    H5Fclose(file);

    REQUIRE(a_read.size() == 10);
    REQUIRE(b_read.size() == 10);
    for (int i = 0; i < 10; ++i) {
        REQUIRE(a_read[static_cast<std::size_t>(i)] == static_cast<double>(i) + 0.25);
        REQUIRE(b_read[static_cast<std::size_t>(i)] == i * 7);
    }
}

TEST_CASE("Writer::attr round-trips scalar and string attributes", "[io][attr]") {
    reticolo::test::ScratchH5 f{"attr"};

    {
        Writer w{f.path()};
        w.attr<double>("/vars@kappa", 0.137);
        w.attr<int>("/vars@n_md", 20);
        w.attr<std::string>("/vars@note", std::string{"phi4 hot start"});
    }

    hid_t file = H5Fopen(f.path().string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    REQUIRE(file >= 0);
    hid_t vars = H5Gopen2(file, "/vars", H5P_DEFAULT);
    REQUIRE(vars >= 0);

    double kappa = 0.0;
    hid_t a      = H5Aopen(vars, "kappa", H5P_DEFAULT);
    REQUIRE(a >= 0);
    REQUIRE(H5Aread(a, H5T_NATIVE_DOUBLE, &kappa) >= 0);
    H5Aclose(a);

    int n_md = 0;
    a        = H5Aopen(vars, "n_md", H5P_DEFAULT);
    REQUIRE(a >= 0);
    REQUIRE(H5Aread(a, H5T_NATIVE_INT, &n_md) >= 0);
    H5Aclose(a);

    char* note_cstr = nullptr;
    a               = H5Aopen(vars, "note", H5P_DEFAULT);
    REQUIRE(a >= 0);
    hid_t s_type = H5Tcopy(H5T_C_S1);
    H5Tset_size(s_type, H5T_VARIABLE);
    H5Tset_strpad(s_type, H5T_STR_NULLTERM);
    REQUIRE(H5Aread(a, s_type, &note_cstr) >= 0);
    std::string note = note_cstr;
    H5free_memory(note_cstr);
    H5Tclose(s_type);
    H5Aclose(a);

    H5Gclose(vars);
    H5Fclose(file);

    REQUIRE(kappa == 0.137);
    REQUIRE(n_md == 20);
    REQUIRE(note == "phi4 hot start");
}
