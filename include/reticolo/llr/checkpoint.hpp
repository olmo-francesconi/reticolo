#pragma once

#include <reticolo/io/reader.hpp>
#include <reticolo/io/writer.hpp>

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <format>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
    #include <omp.h>
#endif

// Ensemble checkpoint for a full LLR run: ONE HDF5 file holding every piece of
// state a deterministic resume needs — each replica's field + HMC StreamSet +
// adapted `a`, plus the orchestrator's exchange RNG and its position in the
// (phase, iter) schedule. Written at a sweep boundary (after exchange, before
// the next thermalize), so the per-replica action caches need not be saved:
// resume's first hmc.step repopulates them before any energy()/exchange read.
//
// Reproducibility rests on two structural facts (see driver.hpp / parallel.hpp):
// replicas share no RNG during the parallel phase (exch_rng is serial), and
// per-replica HMC draws are bound to the partition ITEM index, not the thread —
// so a threaded replica (m>1) is bit-reproducible for a FIXED (n_threads,
// slabs). The checkpoint records that pair; the StreamSet loader already throws
// on a stream-count mismatch, so a resume with a different m fails loudly rather
// than silently forking the chain.
//
// Layout:
//     /orch@n_rep /orch@phase /orch@iter /orch@n_threads /orch@slabs
//     /orch/exch_rng                         FastRng uint64[4] + cached normal
//     /replica_NNN/field                     1-D field dataset (+ shape attrs)
//     /replica_NNN/rng                        StreamSet (driver + site streams)
//     /replica_NNN@a                          adapted tilt

namespace reticolo::llr {

// Where the driver stopped, so a resume re-enters the exact schedule slot.
// phase: 0 = Newton-Raphson warm-up, 1 = Robbins-Monro. iter: the NEXT
// iteration index to run within that phase (k for NR, s for RM). n_threads /
// slabs: the per-replica HMC threading the file was written under — resume
// must reconstruct replicas with the same, or the StreamSet load rejects it.
struct OrchState {
    int phase     = 0;
    int iter      = 0;
    int n_threads = 1;
    int slabs     = 0;
};

// Allocate `total_threads` over `n_rep` replicas. Replica parallelism is ~linear
// and bandwidth-efficient; site-level HMC threading is sublinear and BW-capped
// (~6×), so saturate replicas first and spill leftover threads into per-replica
// site teams only when replicas run out — never past the site-scaling knee
// `m_max`. requested_m: >0 pins m; 0 = auto (m = clamp(T/n_rep, 1, m_max));
// <0 or 1 keeps the serial default (m=1, today's flat model). concurrency is the
// outer replica-team size, m·concurrency ≤ T.
struct ThreadPlan {
    int m           = 1;  // HMC threads per replica
    int concurrency = 0;  // replicas run at once (outer omp team); 0 = ambient
};

[[nodiscard]] inline ThreadPlan
plan_threads(int n_rep, int total_threads, int requested_m, int m_max = 4) {
    int total = total_threads;
#ifdef _OPENMP
    if (total <= 0) {
        total = omp_get_max_threads();
    }
#endif
    if (total <= 0) {
        total = 1;
    }
    int m = 1;
    if (requested_m > 0) {
        m = requested_m;
    } else if (requested_m == 0) {
        m = std::clamp(total / std::max(n_rep, 1), 1, m_max);
    }
    int const conc = std::clamp(total / std::max(m, 1), 1, std::max(n_rep, 1));
    return {.m = m, .concurrency = conc};
}

// Atomic overwrite: write the snapshot to `path`.tmp, then rename over `path`,
// so a crash mid-write never truncates the live checkpoint. reps is non-const —
// StreamSet::state_words() and phi() are read through the replica's mutable
// accessors.
template <class Replica, class ExchRng>
void save_ensemble(std::filesystem::path const& path,
                   std::vector<std::unique_ptr<Replica>>& reps,
                   ExchRng const& exch_rng,
                   OrchState const& orch) {
    namespace fs           = std::filesystem;
    fs::path tmp           = path;
    tmp                    += ".tmp";
    auto const n_rep       = static_cast<int>(reps.size());
    {
        io::Writer w{tmp};
        w.attr<int>("/orch@n_rep", n_rep);
        w.attr<int>("/orch@phase", orch.phase);
        w.attr<int>("/orch@iter", orch.iter);
        w.attr<int>("/orch@n_threads", orch.n_threads);
        w.attr<int>("/orch@slabs", orch.slabs);
        w.rng_state("/orch/exch_rng", exch_rng);
        for (int n = 0; n < n_rep; ++n) {
            auto& rep            = *reps[static_cast<std::size_t>(n)];
            std::string const g  = std::format("/replica_{:03d}", n);
            auto& ss             = rep.rng();
            using SS             = std::remove_reference_t<decltype(ss)>;
            using R              = SS::rng_type;
            w.field(g + "/field", rep.phi());
            w.rng_streams(
                g + "/rng", R::name, ss.state_words(), ss.n_streams(), SS::words_per_stream);
            w.attr<double>(g + "@a", static_cast<double>(rep.a()));
        }
    }
    fs::rename(tmp, path);
}

// Restore every replica's field / StreamSet / a and the exchange RNG, returning
// the schedule position. The replica vector must already be constructed with the
// matching shapes and per-replica threading (its StreamSets are sized from the
// HmcSpec.n_threads passed at construction); the field and rng_streams reads
// validate shape and stream count and throw on any mismatch.
template <class Replica, class ExchRng>
[[nodiscard]] OrchState load_ensemble(std::filesystem::path const& path,
                                      std::vector<std::unique_ptr<Replica>>& reps,
                                      ExchRng& exch_rng) {
    io::Reader r{path};
    OrchState orch{.phase     = r.attr<int>("/orch@phase"),
                   .iter      = r.attr<int>("/orch@iter"),
                   .n_threads = r.attr<int>("/orch@n_threads"),
                   .slabs     = r.attr<int>("/orch@slabs")};
    auto const saved_n_rep = r.attr<int>("/orch@n_rep");
    if (saved_n_rep != static_cast<int>(reps.size())) {
        throw std::runtime_error{
            std::format("llr::load_ensemble: checkpoint has {} replicas, run has {}",
                        saved_n_rep,
                        reps.size())};
    }
    exch_rng = r.rng_state("/orch/exch_rng");
    for (int n = 0; n < saved_n_rep; ++n) {
        auto& rep           = *reps[static_cast<std::size_t>(n)];
        std::string const g = std::format("/replica_{:03d}", n);
        auto& ss            = rep.rng();
        using SS            = std::remove_reference_t<decltype(ss)>;
        using R             = SS::rng_type;
        r.field(g + "/field", rep.phi());
        ss.restore_state_words(
            r.rng_streams(g + "/rng", R::name, ss.n_streams(), SS::words_per_stream));
        rep.set_a(static_cast<Replica::scalar_t>(r.attr<double>(g + "@a")));
    }
    return orch;
}

}  // namespace reticolo::llr
