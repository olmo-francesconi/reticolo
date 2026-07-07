#pragma once

#include <reticolo/core/rng/rng.hpp>
#include <reticolo/io/reader.hpp>
#include <reticolo/io/writer.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <vector>

// Config snapshot — one HDF5 file containing everything an app needs to
// resume from a given trajectory: the field buffer, the RNG state, the next
// trajectory index, and (free of charge) every CLI flag the run was launched
// with, stamped under /vars@* by the Writer/Parser pair.
//
// Layout produced by `save_config`:
//
//     /run@*       Writer reproducibility metadata
//     /vars@*      Parser-resolved flags (when a Parser is passed)
//     /field       1-D dataset + attrs (kind/scalar_type/shape/n_components)
//     /rng         uint64[4] + attrs (kind/cached_normal/has_cached_normal)
//     /traj@i      next trajectory index (long long)
//
// `load_config` is the symmetric reader: open the file, refill the lattice,
// rebuild the RNG, return the trajectory counter. The caller constructs the
// lattice on the shape from `Reader::field_shape("/field")` before calling.

namespace reticolo::cli {
class Parser;
}  // namespace reticolo::cli

namespace reticolo::io {

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): each arg is distinctly named.
template <class Field>
void save_config(std::filesystem::path const& path,
                 Field const& lat,
                 FastRng const& rng,
                 long long traj_i,
                 int argc                = 0,
                 char const* const* argv = nullptr,
                 cli::Parser const* p    = nullptr) {
    Writer w{path, argc, argv, p};
    w.field("/field", lat);
    w.rng_state("/rng", rng);
    w.attr<long long>("/traj@i", traj_i);
}

template <class Field>
[[nodiscard]] long long load_config(std::filesystem::path const& path, Field& lat, FastRng& rng) {
    Reader r{path};
    r.field("/field", lat);
    rng = r.rng_state("/rng");
    return r.attr<long long>("/traj@i");
}

[[nodiscard]] inline std::vector<std::size_t> load_field_shape(std::filesystem::path const& path) {
    Reader r{path};
    return r.field_shape("/field");
}

// Counter-based variant for backends whose RNG is a counter-based generator
// (Philox) rather than a stateful FastRng: the complete per-run RNG state is
// (seed, trajectory counter). Used by the CUDA HmcEngine, whose device RNG is
// not a FastRng — so the FastRng `/rng` uint64[4] layout does not apply. Layout:
//
//     /field        same as save_config
//     /rng@seed      uint64  Philox key
//     /rng@counter   uint64  trajectory counter (next Philox counter on resume)
//     /traj@i        next trajectory/block index (long long)
//
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters): each arg is distinctly named.
template <class Field>
void save_config_counter(std::filesystem::path const& path,
                         Field const& lat,
                         std::uint64_t seed,
                         std::uint64_t counter,
                         long long traj_i,
                         int argc                = 0,
                         char const* const* argv = nullptr,
                         cli::Parser const* p    = nullptr) {
    Writer w{path, argc, argv, p};
    w.field("/field", lat);
    w.attr<unsigned long long>("/rng@seed", seed);
    w.attr<unsigned long long>("/rng@counter", counter);
    w.attr<long long>("/traj@i", traj_i);
}

template <class Field>
[[nodiscard]] long long load_config_counter(std::filesystem::path const& path,
                                            Field& lat,
                                            std::uint64_t& seed,
                                            std::uint64_t& counter) {
    Reader r{path};
    r.field("/field", lat);
    seed    = r.attr<unsigned long long>("/rng@seed");
    counter = r.attr<unsigned long long>("/rng@counter");
    return r.attr<long long>("/traj@i");
}

}  // namespace reticolo::io
