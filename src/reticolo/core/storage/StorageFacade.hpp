#pragma once

#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

#include "reticolo/core/storage/Hdf5StorageBackend.hpp"
#include "reticolo/core/storage/StorageBackendConcept.hpp"
#include "reticolo/core/storage/StorageSchema.hpp"
#include "reticolo/lattice/lattice.hpp"

namespace reticolo::storage {

namespace fs = std::filesystem;

template <StorageBackend Backend>
class StorageManager {
  public:
    StorageManager() = default;

    void               initialize_file(const fs::path& file_name) { _Backend.initFile(file_name); }
    [[nodiscard]] auto file_exists(const fs::path& file_name) -> bool { return _Backend.checkFile(file_name); }

    void ensure_group(const fs::path& file_name, const std::string& group_name) {
        _Backend.createGroup(file_name, group_name);
    }
    void ensure_group(const fs::path& file_name, const schema::ObjectPath& group_path) {
        ensure_group(file_name, group_path.value);
    }

    template <typename T>
    void write_dataset(const fs::path& file_name, const std::string& dataset_name, const std::vector<T>& data) {
        _Backend.template writeDataset<T>(file_name, dataset_name, data);
    }
    template <typename T>
    void write_dataset(const fs::path& file_name, const schema::ObjectPath& dataset_path, const std::vector<T>& data) {
        write_dataset(file_name, dataset_path.value, data);
    }

    template <typename T>
    void setup_appendable_dataset(const fs::path& file_name, const std::string& dataset_name, hsize_t chunk_size,
                                  bool compressed) {
        _Backend.template setupExpandableDataset<T>(file_name, dataset_name, chunk_size, compressed);
    }
    template <typename T>
    void setup_appendable_dataset(const fs::path& file_name, const schema::AppendableDatasetSpec& spec) {
        setup_appendable_dataset<T>(file_name, spec.path.value, spec.chunk_size, spec.compressed);
    }

    template <typename T>
    void append_dataset(const fs::path& file_name, const std::string& dataset_name, const std::vector<T>& data) {
        _Backend.template appendDataset<T>(file_name, dataset_name, data);
    }
    template <typename T>
    void append_dataset(const fs::path& file_name, const schema::ObjectPath& dataset_path, const std::vector<T>& data) {
        append_dataset(file_name, dataset_path.value, data);
    }

    template <typename T>
    void save_lattice(const fs::path& file_name, const std::string& lattice_id, const Lattice<T>& field,
                      const std::stringstream& rng_state) {
        _Backend.template saveLattice<T>(file_name, lattice_id, field, rng_state);
    }
    template <typename T>
    void save_lattice(const fs::path& file_name, const schema::ObjectPath& lattice_path, const Lattice<T>& field,
                      const std::stringstream& rng_state) {
        save_lattice(file_name, lattice_path.value, field, rng_state);
    }

    template <typename T>
    void load_lattice(const fs::path& file_name, const std::string& lattice_id, Lattice<T>& field,
                      std::stringstream& rng_state) {
        _Backend.template readLattice<T>(file_name, lattice_id, field, rng_state);
    }
    template <typename T>
    void load_lattice(const fs::path& file_name, const schema::ObjectPath& lattice_path, Lattice<T>& field,
                      std::stringstream& rng_state) {
        load_lattice(file_name, lattice_path.value, field, rng_state);
    }

    // Compatibility wrappers for the pre-facade naming.
    void               initFile(const fs::path& file_name) { initialize_file(file_name); }
    [[nodiscard]] auto checkFile(const fs::path& file_name) -> bool { return file_exists(file_name); }
    void createGroup(const fs::path& file_name, const std::string& group_name) { ensure_group(file_name, group_name); }

    template <typename T>
    void writeDataset(const fs::path& file_name, const std::string& dataset_name, const std::vector<T>& data) {
        write_dataset(file_name, dataset_name, data);
    }

    template <typename T>
    void setupExpandableDataset(const fs::path& file_name, const std::string& dataset_name, hsize_t chunk_size,
                                bool compressed) {
        setup_appendable_dataset<T>(file_name, dataset_name, chunk_size, compressed);
    }

    template <typename T>
    void appendDataset(const fs::path& file_name, const std::string& dataset_name, const std::vector<T>& data) {
        append_dataset(file_name, dataset_name, data);
    }

    template <typename T>
    void saveLattice(const fs::path& file_name, const std::string& lattice_id, const Lattice<T>& field,
                     const std::stringstream& rng_state) {
        save_lattice(file_name, lattice_id, field, rng_state);
    }

    template <typename T>
    void readLattice(const fs::path& file_name, const std::string& lattice_id, Lattice<T>& field,
                     std::stringstream& rng_state) {
        load_lattice(file_name, lattice_id, field, rng_state);
    }

  private:
    Backend _Backend;
};

using DefaultStorageBackend = Hdf5StorageBackend;
using Storage = StorageManager<DefaultStorageBackend>;

inline Storage GlobalStorage;

}  // namespace reticolo::storage
