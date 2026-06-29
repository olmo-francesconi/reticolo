#pragma once

#include <reticolo/cli/parser.hpp>
#include <reticolo/core/log.hpp>
#include <reticolo/io/writer.hpp>

#include <filesystem>
#include <string>

// Setup-only helpers shared by the reference apps under `apps/`. They absorb the
// pre-loop scaffolding that is byte-for-byte identical across every app — the
// truly-universal flag block and the workspace/writer open — so it can't drift
// (the resume flag was once added to phi4_hmc only; the shared block keeps the
// common flag names/descriptions in one place).
//
// Deliberately minimal. Only flags that are present with the SAME description in
// every app live here: `L,size`, `seed`, `workspace`, `out`. Everything else is
// app-specific and stays in the app — `ndim` (absent for the 2D-only XY model),
// the physics couplings, and the trajectory counts (whose wording is
// app-appropriate: "trajectories" for HMC, "sweeps" for Metropolis,
// "measurements" for Wolff, NR/RM counts for LLR). And the trajectory `for` loop
// always stays in the app: these helpers never drive it.

namespace reticolo::app {

// Per-app values for the universal flag block. Flag names, types, and help text
// are single-sourced in `common_flags`; only `L` and `out` vary enough to pass
// (seed and workspace are the same everywhere, overridable if ever needed).
struct CommonDefaults {
    int L                   = 8;
    unsigned long long seed = 42ULL;
    std::string out         = "out.h5";
};

// Stable references to the universal flags, valid after `parser.parse()`.
struct CommonFlags {
    int const& L;
    unsigned long long const& seed;
    std::string const& workspace;
    std::string const& out;
};

// Register the flag block every reference app shares onto `p`, returning stable
// references to read after `p.parse()`. Apps register `ndim` / physics / loop
// counts themselves, before or after this call.
inline CommonFlags common_flags(cli::Parser& p, CommonDefaults const& d) {
    int const& L                   = p.opt<int>("L,size", d.L, "linear lattice extent");
    unsigned long long const& seed = p.opt<unsigned long long>("seed", d.seed, "RNG seed");
    std::string const& workspace =
        p.opt<std::string>("workspace", std::string{"."}, "workspace folder (output + logs)");
    std::string const& out =
        p.opt<std::string>("out", d.out, "HDF5 output file name, inside workspace");
    return CommonFlags{L, seed, workspace, out};
}

// The output path an app writes to: `<workspace>/<out>`. Exposed for apps that
// also need it directly (e.g. per-config checkpoint file names).
[[nodiscard]] inline std::string out_path(CommonFlags const& f) {
    return (std::filesystem::path{f.workspace} / f.out).string();
}

// Open the workspace log + HDF5 writer the way every app does:
// `log::start(workspace, out[, replicas])` then `Writer{<workspace>/<out>, …}`.
// LLR apps pass `replicas = true`. Returns the move-only Writer by value; the
// app starts its own phases/series and owns its loop.
[[nodiscard]] inline io::Writer open_writer(
    cli::Parser const& p, CommonFlags const& f, int argc, char** argv, bool replicas = false) {
    log::start(f.workspace, f.out, replicas);
    return io::Writer{out_path(f), argc, argv, &p};
}

}  // namespace reticolo::app
