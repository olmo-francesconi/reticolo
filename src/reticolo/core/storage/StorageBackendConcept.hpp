#pragma once

#include <concepts>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "reticolo/lattice/lattice.hpp"

namespace reticolo::storage {

template <typename Backend>
concept StorageBackend = requires(
    Backend backend, const std::filesystem::path& file_name, const std::string& object_name,
    const std::vector<double>& data, Lattice<double>& lattice, const Lattice<double>& const_lattice,
    std::stringstream& rng_state, const std::stringstream& const_rng_state, hsize_t chunk_size, bool compressed) {
    { backend.initFile(file_name) } -> std::same_as<void>;
    { backend.checkFile(file_name) } -> std::same_as<bool>;
    { backend.createGroup(file_name, object_name) } -> std::same_as<void>;
    { backend.writeDataset(file_name, object_name, data) } -> std::same_as<void>;
    {
        backend.template setupExpandableDataset<double>(file_name, object_name, chunk_size, compressed)
    } -> std::same_as<void>;
    { backend.appendDataset(file_name, object_name, data) } -> std::same_as<void>;
    { backend.saveLattice(file_name, object_name, const_lattice, const_rng_state) } -> std::same_as<void>;
    { backend.readLattice(file_name, object_name, lattice, rng_state) } -> std::same_as<void>;
};

}  // namespace reticolo::storage
