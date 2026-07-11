#pragma once

#include <reticolo/io/reader.hpp>
#include <reticolo/io/writer.hpp>
#include <reticolo/orch/concepts.hpp>

#include <cstddef>
#include <filesystem>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace reticolo::orch {

// Generic ensemble checkpoint: ONE HDF5 file holding every worker's config
// buffer + owned RNG, plus the orchestrator's schedule position. Physics-free —
// the spine snapshots any `Checkpointable` worker (a span Chain, an LLR
// Replica). Two extension points keep it that way:
//   * per-worker `save_extra` / `load_extra` (opt-in, `HasCheckpointExtra`) for
//     workflow-specific per-worker payload (an LLR replica's tilt `a`);
//   * the `orch_extra` callable for orchestrator-level state (an LLR exchange
//     RNG) — invoked once, after the /orch header.
//
// Layout (prefix = "member" by default; LLR passes "replica" to preserve its
// historical /replica_NNN paths):
//     /orch@n_rep /orch@phase /orch@iter /orch@n_threads /orch@slabs
//     <orch_extra output, e.g. /orch/exch_rng>
//     /<prefix>_NNN/field        config dataset (+ shape attrs)
//     /<prefix>_NNN/rng          StreamSet (driver + site streams)
//     <per-worker save_extra output, e.g. /<prefix>_NNN@a>

// Where an orchestrator stopped, so a resume re-enters the exact schedule slot.
// The meaning of `phase` / `iter` is the workflow's (for LLR: phase 0 = NR,
// 1 = RM; iter = next index within it). n_threads / slabs record the per-worker
// HMC threading the file was written under — a resume must reconstruct workers
// with the same, or the StreamSet load rejects the stream count.
struct OrchState {
    int phase     = 0;
    int iter      = 0;
    int n_threads = 1;
    int slabs     = 0;
};

// Opt-in per-worker checkpoint payload beyond field + rng.
template <class W>
concept HasCheckpointExtra =
    requires(W& w, io::Writer& out, io::Reader& in, std::string const& g) {
        w.save_extra(out, g);
        w.load_extra(in, g);
    };

// Atomic overwrite: write to `path`.tmp then rename over `path`, so a crash
// mid-write never truncates the live checkpoint. `workers` is non-const —
// StreamSet::state_words() and field() are read through mutable accessors.
template <class Worker, class OrchExtra>
    requires Checkpointable<Worker>
void save_ensemble(std::filesystem::path const& path,
                   std::vector<std::unique_ptr<Worker>>& workers,
                   OrchState const& state,
                   std::string_view prefix,
                   OrchExtra&& orch_extra) {
    namespace fs         = std::filesystem;
    fs::path tmp         = path;
    tmp                 += ".tmp";
    auto const n_workers = static_cast<int>(workers.size());
    {
        io::Writer w{tmp};
        w.attr<int>("/orch@n_rep", n_workers);
        w.attr<int>("/orch@phase", state.phase);
        w.attr<int>("/orch@iter", state.iter);
        w.attr<int>("/orch@n_threads", state.n_threads);
        w.attr<int>("/orch@slabs", state.slabs);
        orch_extra(w);
        for (int n = 0; n < n_workers; ++n) {
            auto& wk            = *workers[static_cast<std::size_t>(n)];
            std::string const g = std::format("/{}_{:03d}", prefix, n);
            auto& ss            = wk.rng();
            using SS            = std::remove_reference_t<decltype(ss)>;
            using R             = SS::rng_type;
            w.field(g + "/field", wk.field());
            w.rng_streams(
                g + "/rng", R::name, ss.state_words(), ss.n_streams(), SS::words_per_stream);
            if constexpr (HasCheckpointExtra<Worker>) {
                wk.save_extra(w, g);
            }
        }
    }
    fs::rename(tmp, path);
}

// Restore every worker's field / StreamSet (+ save_extra payload) and the
// orchestrator extra, returning the schedule position. The worker vector must
// already be constructed with matching shapes and per-worker threading (its
// StreamSets are sized from the HmcSpec passed at construction); the field and
// rng_streams reads validate shape and stream count and throw on any mismatch.
template <class Worker, class OrchExtra>
    requires Checkpointable<Worker>
[[nodiscard]] OrchState load_ensemble(std::filesystem::path const& path,
                                      std::vector<std::unique_ptr<Worker>>& workers,
                                      std::string_view prefix,
                                      OrchExtra&& orch_extra) {
    io::Reader r{path};
    OrchState state{.phase     = r.attr<int>("/orch@phase"),
                    .iter      = r.attr<int>("/orch@iter"),
                    .n_threads = r.attr<int>("/orch@n_threads"),
                    .slabs     = r.attr<int>("/orch@slabs")};
    auto const saved = r.attr<int>("/orch@n_rep");
    if (saved != static_cast<int>(workers.size())) {
        throw std::runtime_error{std::format(
            "orch::load_ensemble: checkpoint has {} workers, run has {}", saved, workers.size())};
    }
    orch_extra(r);
    for (int n = 0; n < saved; ++n) {
        auto& wk            = *workers[static_cast<std::size_t>(n)];
        std::string const g = std::format("/{}_{:03d}", prefix, n);
        auto& ss            = wk.rng();
        using SS            = std::remove_reference_t<decltype(ss)>;
        using R             = SS::rng_type;
        r.field(g + "/field", wk.field());
        ss.restore_state_words(
            r.rng_streams(g + "/rng", R::name, ss.n_streams(), SS::words_per_stream));
        if constexpr (HasCheckpointExtra<Worker>) {
            wk.load_extra(r, g);
        }
    }
    return state;
}

}  // namespace reticolo::orch
