#include <H5Dpublic.h>
#include <H5Fpublic.h>
#include <H5Ppublic.h>
#include <H5Spublic.h>
#include <H5Tpublic.h>

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

#include "reticolo/core/storage/StorageFacade.hpp"
#include "reticolo/core/storage/StorageSchema.hpp"

namespace fs = std::filesystem;

static auto make_tmp_h5_path() -> fs::path {
    std::random_device                                rd;
    std::mt19937_64                                   gen(rd());
    std::uniform_int_distribution<unsigned long long> dis;

    for (int attempt = 0; attempt < 16; attempt++) {
        const auto      suffix = std::to_string(dis(gen));
        fs::path        p = fs::temp_directory_path() / ("reticolo_hdf5_basic_" + suffix + ".h5");
        std::error_code ec;
        if (!fs::exists(p, ec)) {
            return p;
        }
    }
    // Extremely unlikely, but fail loudly
    throw std::runtime_error("Failed to generate unique temp .h5 path");
}

static void require(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "TEST FAILED: %s\n", msg);
        std::abort();
    }
}

auto main() -> int {
    const fs::path file = make_tmp_h5_path();

    try {
        const auto group_path = reticolo::storage::schema::object("g");
        const auto dataset_path = reticolo::storage::schema::object("g/data");
        reticolo::storage::GlobalStorage.initialize_file(file);

        // Group creation should be idempotent
        reticolo::storage::GlobalStorage.ensure_group(file, group_path);
        reticolo::storage::GlobalStorage.ensure_group(file, group_path);

        // writeDataset should overwrite if dataset exists
        reticolo::storage::GlobalStorage.write_dataset<double>(file, dataset_path, std::vector<double>{1.0, 2.0, 3.0});
        reticolo::storage::GlobalStorage.write_dataset<double>(file, dataset_path, std::vector<double>{4.0, 5.0});

        // Read back using raw HDF5 API
        hid_t f = H5Fopen(file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        require(f >= 0, "H5Fopen failed");

        hid_t d = H5Dopen2(f, "g/data", H5P_DEFAULT);
        require(d >= 0, "H5Dopen2 failed");

        hid_t space = H5Dget_space(d);
        require(space >= 0, "H5Dget_space failed");

        hsize_t dims[1] = {0};
        require(H5Sget_simple_extent_ndims(space) == 1, "dataset dims != 1");
        require(H5Sget_simple_extent_dims(space, dims, nullptr) >= 0, "H5Sget_simple_extent_dims failed");
        require(dims[0] == 2, "dataset size mismatch after overwrite");

        std::vector<double> out(dims[0], 0.0);
        require(H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()) >= 0, "H5Dread failed");
        require(out.size() == 2 && out[0] == 4.0 && out[1] == 5.0, "dataset contents mismatch after overwrite");

        H5Sclose(space);
        H5Dclose(d);
        H5Fclose(f);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Exception: %s\n", e.what());
        (void)std::remove(file.c_str());
        return 1;
    }

    (void)std::remove(file.c_str());
    return 0;
}
