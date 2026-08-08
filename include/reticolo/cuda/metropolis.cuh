#pragma once

// Generic device Metropolis — nvcc-only (.cuh; transitively launches kernels).
//
// Mirrors updater::Metropolis::step() once, the way cuda::Hmc mirrors
// updater::Hmc::step(). There is no action switch: the action is any type
// exposing metropolis_stencil (here cuda::DeviceAction, delegating to the action's
// device_functors trait), and the sweep is the generic checkerboard skeleton.
//
// HOST-FREE SWEEPS, for the same reason and by the same mechanism as the HMC
// trajectory graph: the whole sweep — both colour kernels, both reductions, the
// counter bump — is captured into ONE cudaGraph and replayed with a single
// cudaGraphLaunch. Nothing returns to the host between sweeps. The accept test
// is already per-site on the device (there is no global accept/reject to hand
// back), so unlike HMC there is not even a decision kernel: a Metropolis sweep is
// structurally the more graph-friendly of the two.
//
// The capture trap (graph.hpp) applies unchanged: the per-sweep-varying input is
// the Philox counter, which is read from a device buffer and bumped on the
// device, never baked as a kernel literal.
//
// COLOURS come from the action, not from here: 2 for a site field, 2*ndim for a
// link field, whose elementary move is per (direction, site parity). The driver
// only loops `action.n_colors()` — the same opacity the host updater has.
//
// STATE. Far less than HMC: no momentum, no force, no rollback copy. The two
// site-indexed double buffers below are diagnostics, not physics — they stage the
// per-site ΔS and accept flag for the ordinary reduce, keeping the sweep kernel
// free of atomics (the gather-only invariant) and of a block reduction.
// Folding a block reduction into the kernel would retire both; worth doing once
// the sweep is shown to be bandwidth-bound rather than gather-latency-bound.

#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/checkerboard.cuh>
#include <reticolo/cuda/device_action.cuh>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/graph.hpp>
#include <reticolo/cuda/reduce.cuh>
#include <reticolo/cuda/stream.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>

#include <cuda_runtime.h>

namespace reticolo::cuda {

// Templated for weak/mergeable linkage, as every other header kernel here.
template <class Dummy = void>
__global__ void bump_sweep_kernel(std::uint64_t* sweep) {
    ++(*sweep);
}

// Carry the sweep's ΣΔS into the running S and accumulate the acceptance, all on
// the device so a run of k sweeps needs no readback. `stats` = {S, accepted,
// attempts}; `red` = {Σ ds, Σ accepted-hits} from the two reductions.
template <class Dummy = void>
__global__ void apply_sweep_stats_kernel(double* stats, double const* red, double attempts) {
    stats[0] += red[0];
    stats[1] += red[1];
    stats[2] += attempts;
}

struct MetropolisResult {
    std::size_t n_accepted = 0;
    std::size_t n_attempts = 0;
    double ds              = 0.0;

    [[nodiscard]] double acceptance() const noexcept {
        return n_attempts == 0 ? 0.0
                               : static_cast<double>(n_accepted) / static_cast<double>(n_attempts);
    }
};

template <class A, class Field = DeviceField<double>>
class Metropolis {
public:
    Metropolis(A action, Field& field, double sigma, std::uint64_t seed = 0xC0FFEEULL)
        : action_{std::move(action)}, field_{field}, sigma_{sigma}, seed_{seed}, sweep_buf_{1},
          ds_site_{field.size()}, acc_site_{field.size()},
          partials_{static_cast<std::size_t>(k_reduce_max_grid)}, red_{2}, stats_{3},
          stream_{make_stream_()}, graph_{stream_} {
        validate_checkerboard_();
        ScopedStream const scope{stream_};
        std::uint64_t const zero = 0;
        sweep_buf_.copy_from_host(&zero, stream_);
        // stats_ = {S, accepted, attempts}; S is seeded from a real s_full so the
        // incremental carry below starts from the truth.
        std::array<double, 3> const init{action_.s_full(field_), 0, 0};
        stats_.copy_from_host(init.data(), stream_);
        RETICOLO_CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

    ~Metropolis() { cudaStreamDestroy(stream_); }

    Metropolis(Metropolis const&)            = delete;
    Metropolis& operator=(Metropolis const&) = delete;
    Metropolis(Metropolis&&)                 = delete;
    Metropolis& operator=(Metropolis&&)      = delete;

    // Replay k full sweeps host-free: one cudaGraphLaunch each, no sync, no
    // readback. Sync only to measure.
    void run(int k) {
        ScopedStream const scope{stream_};
        for (int i = 0; i < k; ++i) {
            graph_.run([&] { enqueue_sweep_(); });
        }
        sweep_count_ += static_cast<std::uint64_t>(k);
    }

    void sync() { RETICOLO_CUDA_CHECK(cudaStreamSynchronize(stream_)); }

    // One sweep with host-visible diagnostics (the per-step probe path): replay
    // once, read the deltas back, sync once. `run(k)` is the fast path.
    MetropolisResult step() {
        auto const before = read_stats_();
        run(1);
        auto const st = read_stats_();
        return {.n_accepted = static_cast<std::size_t>(st[1] - before[1]),
                .n_attempts = static_cast<std::size_t>(st[2] - before[2]),
                .ds         = st[0] - before[0]};
    }

    // S of the current config, carried incrementally on the device across every
    // sweep — the device-resident twin of updater::Metropolis::last_s_full().
    // Syncs.
    [[nodiscard]] double last_s_full() { return read_stats_()[0]; }

    // Re-derive S with a full reduction, discarding the accumulated rounding.
    double resync_s_full() {
        ScopedStream const scope{stream_};
        double const s = action_.s_full(field_);
        auto st        = read_stats_();
        st[0]          = s;
        stats_.copy_from_host(st.data(), stream_);
        sync();
        return s;
    }

    [[nodiscard]] double acceptance() {
        auto const st = read_stats_();
        return st[2] == 0.0 ? 0.0 : st[1] / st[2];
    }

    [[nodiscard]] A& action() noexcept { return action_; }
    [[nodiscard]] A const& action() const noexcept { return action_; }

    // --- checkpoint hooks ---------------------------------------------------
    // Same contract as cuda::Hmc: the complete per-run RNG state is (seed_,
    // device sweep counter), because Philox is stateless given (seed, counter,
    // pair). Field + these two reproduce an uninterrupted run bit-for-bit.
    [[nodiscard]] std::uint64_t seed() const { return seed_; }

    [[nodiscard]] std::uint64_t rng_counter() {
        std::uint64_t c = 0;
        sweep_buf_.copy_to_host(&c, stream_);
        sync();
        return c;
    }
    void set_rng_counter(std::uint64_t c) {
        sweep_buf_.copy_from_host(&c, stream_);
        sync();
    }

    void set_sigma(double s) {
        if (s != sigma_) {
            // sigma is captured as a kernel argument, so wait for any replay
            // using the old value before dropping its graph executable.
            sync();
            sigma_ = s;
            graph_.invalidate();
        }
    }
    [[nodiscard]] double sigma() const noexcept { return sigma_; }

    // Start a new acceptance-reporting phase without perturbing the carried
    // action or Philox state. Apps call this at the therm→prod boundary.
    void reset_acceptance_stats() {
        auto st = read_stats_();
        st[1]   = 0.0;
        st[2]   = 0.0;
        stats_.copy_from_host(st.data(), stream_);
        sync();
    }

private:
    void validate_checkerboard_() const {
        auto const& topo = field_.topology();
        for (int mu = 0; mu < topo.ndim; ++mu) {
            if ((topo.shape[mu] % 2) != 0) {
                throw std::invalid_argument{
                    "cuda::Metropolis: the checkerboard sweep needs every lattice extent even"};
            }
        }
    }

    [[nodiscard]] static cudaStream_t make_stream_() {
        cudaStream_t s = nullptr;
        RETICOLO_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        return s;
    }

    // The full per-sweep operation sequence, captured once and replayed. Note
    // sigma_ is baked into the capture. `set_sigma()` synchronizes and invalidates
    // the cached graph before the next replay, so it remains correct but is not a
    // hot-path operation.
    void enqueue_sweep_() {
        // The per-site staging buffers are indexed by SITE, so the reduction
        // length is nsites for both families — a link colour updates the links of
        // one direction on one parity, i.e. nsites/2 of them, still site-indexed.
        long const n         = action_.nsites();
        int const n_colors   = action_.n_colors();
        auto const per_color = static_cast<double>(n / 2);
        for (int color = 0; color < n_colors; ++color) {
            action_.metropolis_stencil(field_,
                                       color,
                                       sigma_,
                                       seed_,
                                       sweep_buf_.data(),
                                       ds_site_.data(),
                                       acc_site_.data(),
                                       stream_);
            reduce_sum_into(red_.data() + 0, ds_site_.data(), n, partials_.data(), stream_);
            reduce_sum_into(red_.data() + 1, acc_site_.data(), n, partials_.data(), stream_);
            apply_sweep_stats_kernel<void>
                <<<1, 1, 0, stream_>>>(stats_.data(), red_.data(), static_cast<double>(n / 2));
            RETICOLO_CUDA_CHECK_LAUNCH();
        }
        bump_sweep_kernel<void><<<1, 1, 0, stream_>>>(sweep_buf_.data());
        RETICOLO_CUDA_CHECK_LAUNCH();
    }

    [[nodiscard]] std::array<double, 3> read_stats_() {
        std::array<double, 3> h{};
        stats_.copy_to_host(h.data(), stream_);
        sync();
        return h;
    }

    A action_;
    Field& field_;
    double sigma_;
    std::uint64_t seed_;
    std::uint64_t sweep_count_ = 0;
    DeviceBuffer<std::uint64_t> sweep_buf_;  // device sweep counter (Philox key)
    DeviceBuffer<double> ds_site_;           // per-site applied ΔS (staged for the reduce)
    DeviceBuffer<double> acc_site_;          // per-site accepted-hit count
    DeviceBuffer<double> partials_;          // reduction scratch (no per-sweep malloc)
    DeviceBuffer<double> red_;               // {Σ ds, Σ accepted} of one colour
    DeviceBuffer<double> stats_;             // {S, accepted, attempts}, carried on device
    cudaStream_t stream_;
    TrajectoryGraph graph_;
};

}  // namespace reticolo::cuda
