#pragma once

#include <reticolo/cli/parser.hpp>
#include <reticolo/core/log/log.hpp>
#include <reticolo/io/checkpoint.hpp>
#include <reticolo/io/writer.hpp>

#include <filesystem>
#include <stdexcept>
#include <string>

// Setup-only helpers shared by the reference apps under `apps/`. They absorb the
// pre-loop scaffolding that is byte-for-byte identical across every app — the
// truly-universal flag block and the workspace/writer open — so it can't drift
// (the resume flag was once added to phi4_hmc only; the shared block keeps the
// common flag names/descriptions in one place).
//
// Two tiers, and nothing else:
//   * `common_flags` — flags every app shares regardless of algorithm:
//     `L,size`, `seed`, `workspace`, `out`.
//   * one helper per ALGORITHM FAMILY, covering the run-control block that
//     family shares verbatim: `llr_run_flags` for LLR, `hmc_run_flags` for HMC.
//     A family helper is all-or-nothing — every app in the family uses it, or it
//     does not exist. (HMC went 18 apps without one, which is how `--resume`
//     came to work in exactly one of them.)
// Physics couplings and `ndim` (absent for the 2D-only XY model) stay in the
// app. And the trajectory `for` loop ALWAYS stays in the app: these helpers set
// up flags and I/O, they never drive it.

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
    int const& L     = p.opt<int>("L,size", d.L, "linear lattice extent");
    auto const& seed = p.opt<unsigned long long>("seed", d.seed, "RNG seed");
    auto const& workspace =
        p.opt<std::string>("workspace", std::string{"."}, "workspace folder (output + logs)");
    auto const& out = p.opt<std::string>("out", d.out, "HDF5 output file name, inside workspace");
    return CommonFlags{.L = L, .seed = seed, .workspace = workspace, .out = out};
}

// The output path an app writes to: `<workspace>/<out>`. Exposed for apps that
// also need it directly (e.g. per-config checkpoint file names).
[[nodiscard]] inline std::string out_path(CommonFlags const& f) {
    return (std::filesystem::path{f.workspace} / f.out).string();
}

// Stable references to the LLR run-control flags: per-replica HMC threading,
// the warm-seating budget, and checkpoint/resume. The total CPU budget is NOT a
// flag — `orch::plan_threads` reads it from the OpenMP environment
// (OMP_NUM_THREADS) and splits it into an outer replica team and inner
// per-replica HMC teams. Single-sourced because every LLR app takes the same
// seven with the same wording; the app feeds `replica_threads` to
// `orch::plan_threads` and the rest into its DriverSpec.
struct LlrRunFlags {
    int const& replica_threads;
    int const& slabs;
    int const& warm_therm;
    int const& warm_max_traj;
    std::string const& resume;
    std::string const& checkpoint;
    int const& checkpoint_every;
};

inline LlrRunFlags llr_run_flags(cli::Parser& p) {
    int const& replica_threads = p.opt<int>(
        "replica_threads", 1, "HMC threads per replica (1 = serial default; 0 = auto-balance)");
    int const& slabs = p.opt<int>("slabs_per_thread", 0, "HMC slab granularity per thread (0 = 1)");
    // The warm phase seats each replica in its window before NR; it dominates
    // short runs (thousands of trajectories per replica against a handful of NR/RM
    // sweeps) and is skipped entirely on --resume. Exposed so smoke tests and
    // tuning runs can bound it; 0 skips it.
    int const& warm_therm =
        p.opt<int>("warm_therm", 200, "base-action thermalisation trajectories before seating");
    int const& warm_max_traj = p.opt<int>(
        "warm_max_traj", 2000, "max seating trajectories per replica (0 = no warm phase)");
    auto const& resume =
        p.opt<std::string>("resume", std::string{}, "resume from an LLR ensemble checkpoint (.h5)");
    auto const& checkpoint = p.opt<std::string>(
        "checkpoint", std::string{}, "rolling ensemble checkpoint path (empty = off)");
    int const& checkpoint_every =
        p.opt<int>("checkpoint_every", 0, "sweeps between checkpoints (0 = only at end)");
    return {.replica_threads  = replica_threads,
            .slabs            = slabs,
            .warm_therm       = warm_therm,
            .warm_max_traj    = warm_max_traj,
            .resume           = resume,
            .checkpoint       = checkpoint,
            .checkpoint_every = checkpoint_every};
}

// Per-app defaults for the HMC run-control block. Only the values vary; the
// flag names and help text are single-sourced below.
struct HmcRunDefaults {
    double tau     = 1.0;
    int n_md       = 20;
    int n_therm    = 200;
    int n_prod     = 1000;
    int meas_every = 1;
};

// Stable references to the HMC run-control flags — the block every `*_hmc` app
// registers identically. `resuming()` is the guard apps branch on for the therm
// phase and the series they open.
struct HmcRunFlags {
    double const& tau;
    int const& n_md;
    int const& n_therm;
    int const& n_prod;
    int const& meas_every;
    int const& checkpoint_every;
    std::string const& resume;

    [[nodiscard]] bool resuming() const noexcept { return !resume.empty(); }
};

inline HmcRunFlags hmc_run_flags(cli::Parser& p, HmcRunDefaults const& d = {}) {
    auto const& tau       = p.opt<double>("tau", d.tau, "HMC trajectory length");
    int const& n_md       = p.opt<int>("n_md", d.n_md, "MD steps per trajectory");
    int const& n_therm    = p.opt<int>("n_therm", d.n_therm, "thermalisation trajectories");
    int const& n_prod     = p.opt<int>("n_prod", d.n_prod, "production trajectories");
    int const& meas_every = p.opt<int>("meas_every", d.meas_every, "measure every N trajectories");
    int const& ckpt_every =
        p.opt<int>("checkpoint_every", 0, "write a config every N prod trajectories (0 = off)");
    auto const& resume =
        p.opt<std::string>("resume", std::string{}, "resume from a previous config (.h5)");
    return {.tau              = tau,
            .n_md             = n_md,
            .n_therm          = n_therm,
            .n_prod           = n_prod,
            .meas_every       = meas_every,
            .checkpoint_every = ckpt_every,
            .resume           = resume};
}

// Resume a single-chain HMC app from a config checkpoint, or start fresh.
// Returns the trajectory index to continue from (0 when not resuming). Loads the
// field AND the updater's RNG state, so a resumed chain is bit-identical to the
// uninterrupted one — which is what the `*_resume_equivalence` tests assert.
// Shape is validated against the app's own `--L`/`--ndim` before anything is
// overwritten.
template <class Field, class Updater, class Shape>
[[nodiscard]] inline long long
resume_or_start(HmcRunFlags const& rf, Field& field, Updater& upd, Shape const& shape) {
    if (!rf.resuming()) {
        return 0;
    }
    if (io::load_field_shape(rf.resume) != shape) {
        throw std::runtime_error{"--resume shape mismatch with --L/--ndim"};
    }
    long long const start_i = io::load_config(rf.resume, field, upd.rng());
    log::info("hmc", "resumed from {} at traj {}", rf.resume, start_i);
    return start_i;
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
