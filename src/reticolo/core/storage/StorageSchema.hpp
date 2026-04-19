#pragma once

#include <H5public.h>

#include <format>
#include <string>
#include <string_view>

namespace reticolo::storage::schema {

struct ObjectPath {
    std::string value;
};

struct AppendableDatasetSpec {
    ObjectPath path;
    hsize_t    chunk_size;
    bool       compressed;
};

inline auto object(std::string path) -> ObjectPath { return ObjectPath{std::move(path)}; }

namespace lattice {

inline constexpr std::string_view field_sizes_attribute = "sizes";
inline constexpr std::string_view rng_state_attribute = "RngState";

inline auto field(std::string_view lattice_id = "field") -> ObjectPath { return object(std::string(lattice_id)); }

}  // namespace lattice

namespace montecarlo {

inline auto run_group(std::string_view run_name) -> ObjectPath { return object(std::string(run_name)); }

inline auto observables_dataset(std::string_view run_name) -> ObjectPath {
    return object(std::format("{}/Observables", run_name));
}

inline auto monte_carlo_dataset(std::string_view run_name) -> ObjectPath {
    return object(std::format("{}/MonteCarlo", run_name));
}

inline auto observables_stream(std::string_view run_name, hsize_t chunk_size, bool compressed = true)
    -> AppendableDatasetSpec {
    return AppendableDatasetSpec{observables_dataset(run_name), chunk_size, compressed};
}

inline auto monte_carlo_stream(std::string_view run_name, hsize_t chunk_size, bool compressed = true)
    -> AppendableDatasetSpec {
    return AppendableDatasetSpec{monte_carlo_dataset(run_name), chunk_size, compressed};
}

}  // namespace montecarlo

namespace llr {

inline auto run_group(std::string_view run_id) -> ObjectPath { return object(std::string(run_id)); }

inline auto ak_history_dataset(std::string_view run_id, unsigned int worker_id) -> ObjectPath {
    return object(std::format("{}/[{}]ak", run_id, worker_id));
}

}  // namespace llr

}  // namespace reticolo::storage::schema
