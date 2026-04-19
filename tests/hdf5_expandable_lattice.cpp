#include <H5Dpublic.h>
#include <H5Fpublic.h>
#include <H5Ppublic.h>
#include <H5Spublic.h>
#include <H5Tpublic.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "reticolo/core/storage/StorageFacade.hpp"
#include "reticolo/core/storage/StorageSchema.hpp"
#include "reticolo/core/types/complex.hpp"
#include "reticolo/lattice/lattice.hpp"

namespace fs = std::filesystem;

static auto make_tmp_h5_path() -> fs::path {
    static unsigned long long counter = 0;
    const auto                now_ns =
        static_cast<unsigned long long>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                            std::chrono::high_resolution_clock::now().time_since_epoch())
                                            .count());
    std::random_device rd;
    const auto         suffix = std::to_string(now_ns) + "_" + std::to_string(rd()) + "_" + std::to_string(counter++);
    return fs::temp_directory_path() / ("reticolo_hdf5_expandable_lattice_" + suffix + ".h5");
}

static void require(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "TEST FAILED: %s\n", msg);
        std::abort();
    }
}

static void test_expandable_dataset(const fs::path& file) {
    const auto group_path = reticolo::storage::schema::object("g");
    const auto dataset_path = reticolo::storage::schema::object("g/arr");
    reticolo::storage::GlobalStorage.ensure_group(file, group_path);

    // setupExpandableDataset should be idempotent (overwrite existing dataset)
    reticolo::storage::GlobalStorage.setup_appendable_dataset<double>(
        file, reticolo::storage::schema::AppendableDatasetSpec{dataset_path, 4, true});
    reticolo::storage::GlobalStorage.setup_appendable_dataset<double>(
        file, reticolo::storage::schema::AppendableDatasetSpec{dataset_path, 4, true});

    reticolo::storage::GlobalStorage.append_dataset<double>(file, dataset_path, std::vector<double>{1.0, 2.0, 3.0});
    reticolo::storage::GlobalStorage.append_dataset<double>(file, dataset_path, std::vector<double>{4.0, 5.0});

    hid_t f = H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    require(f >= 0, "H5Fopen failed");

    hid_t d = H5Dopen2(f, "g/arr", H5P_DEFAULT);
    require(d >= 0, "H5Dopen2 failed");

    hid_t space = H5Dget_space(d);
    require(space >= 0, "H5Dget_space failed");

    hsize_t dims[1] = {0};
    require(H5Sget_simple_extent_ndims(space) == 1, "dataset dims != 1");
    require(H5Sget_simple_extent_dims(space, dims, nullptr) >= 0, "H5Sget_simple_extent_dims failed");
    require(dims[0] == 5, "expandable dataset size mismatch");

    std::vector<double> out(dims[0], 0.0);
    require(H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) >= 0, "H5Dread failed");
    require(out == (std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0}), "expandable dataset contents mismatch");

    H5Sclose(space);
    H5Dclose(d);
    H5Fclose(f);
}

static void test_lattice_roundtrip(const fs::path& file) {
    using reticolo::ComplexD;
    using reticolo::Lattice;

    const std::vector<reticolo::Indexing::size_type> shape = {4, 4, 4, 4};
    Lattice<ComplexD>                                out(shape);
    for (size_t i = 0; i < out.size(); i++) {
        out[i] = ComplexD{static_cast<double>(i), -static_cast<double>(i)};
    }

    std::mt19937_64 rng;
    rng.seed(123);
    // advance a bit to make state non-trivial
    (void)rng();
    (void)rng();

    std::stringstream ss;
    ss << rng;
    const std::string ss_str = ss.str();

    // compute expected next number from the stored state
    std::stringstream ss_check(ss_str);
    std::mt19937_64   rng_check;
    ss_check >> rng_check;
    const auto expected_next = rng_check();

    reticolo::storage::GlobalStorage.save_lattice(file, reticolo::storage::schema::lattice::field(), out, ss);
    reticolo::storage::GlobalStorage.save_lattice(file, reticolo::storage::schema::lattice::field(), out, ss);

    Lattice<ComplexD> in(shape);
    std::stringstream ss_in;
    reticolo::storage::GlobalStorage.load_lattice(file, reticolo::storage::schema::lattice::field(), in, ss_in);

    require(in.size() == out.size(), "lattice size mismatch");
    for (size_t i = 0; i < out.size(); i++) {
        require(in[i] == out[i], "lattice element mismatch");
    }

    std::mt19937_64 rng_in;
    ss_in >> rng_in;
    require(rng_in() == expected_next, "RNG state mismatch after readLattice");
}

auto main() -> int {
    const fs::path file = make_tmp_h5_path();
    try {
        std::fprintf(stderr, "hdf5_expandable_lattice: temp file = %s\n", file.c_str());

        // Ensure we start from a clean slate even if a previous run crashed.
        std::error_code ec;
        if (fs::exists(file, ec) && !ec) {
            std::fprintf(stderr, "hdf5_expandable_lattice: path already exists; attempting remove_all...\n");
            fs::remove_all(file, ec);
            if (ec) {
                std::fprintf(stderr, "hdf5_expandable_lattice: remove_all failed: %s\n", ec.message().c_str());
            }
        }

        if (fs::exists(file, ec) && !ec) {
            std::fprintf(stderr, "hdf5_expandable_lattice: still exists after cleanup; is_dir=%d\n",
                         fs::is_directory(file, ec) ? 1 : 0);
        }

        // Sanity-check that the path is writable with regular IO.
        {
            std::ofstream ofs(file, std::ios::out | std::ios::trunc);
            if (!ofs) {
                std::fprintf(stderr, "hdf5_expandable_lattice: std::ofstream failed to open/truncate\n");
            }
        }
        fs::remove(file, ec);  // remove the placeholder (HDF5 will create the real file)

        reticolo::storage::GlobalStorage.initialize_file(file);
        test_expandable_dataset(file);
        test_lattice_roundtrip(file);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Exception: %s\n", e.what());
        (void)std::remove(file.c_str());
        return 1;
    }

    (void)std::remove(file.c_str());
    return 0;
}
