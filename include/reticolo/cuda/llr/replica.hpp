#pragma once

// One device LLR replica: owns its DeviceField config and a cuda::Hmc whose
// action is the device WindowedAction. The GPU twin of llr::Replica. Non-movable
// (the Hmc holds field_ by reference and bakes device pointers into its graph),
// so drivers hold `vector<unique_ptr<Replica>>` — same as the CPU side.
//
// Unlike the CPU replica there is no `sample(n)` here: the measurement average
// lives in the driver so it can interleave launch-all / gather-all across
// replicas (model B — the streams overlap only if every replica's trajectory is
// enqueued before any sync). The replica exposes the per-trajectory primitives
// launch_trajectory() (async) + read_dE() (gather) instead.

#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/lattice.hpp>
#include <reticolo/cuda/actions/device_functors.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/hmc.cuh>
#include <reticolo/cuda/llr/windowed_action.cuh>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

namespace reticolo::cuda::llr {

template <class HostAction, class Integ = alg::integ::Leapfrog, class Field = DeviceField<double>>
class Replica {
public:
    using value_type = typename Field::value_type;
    using Windowed   = WindowedAction<HostAction, Field>;

    Replica(HostAction host,
            std::vector<std::size_t> const& shape,
            double e_n,
            double delta,
            double tau,
            int n_md,
            std::uint64_t seed,
            double a_init = 0.0)
        : field_{shape},
          hmc_{Windowed{host, field_.topology(), a_init, e_n, delta}, field_, tau, n_md, seed},
          e_n_{e_n}, delta_{delta}, seed_{seed} {
        cold_start();
    }

    Replica(Replica const&)            = delete;
    Replica& operator=(Replica const&) = delete;
    Replica(Replica&&)                 = delete;
    Replica& operator=(Replica&&)      = delete;
    ~Replica()                         = default;

    // Cold config. Scalars / U(1): phi = 0 (memset — 0.0 is the identity there).
    // Matrix groups: the zero matrix is not a group element, so route through the
    // trait's set_cold (identity fill) when it provides one.
    void cold_start() {
        using traits = device_functors<HostAction>;
        if constexpr (requires(cudaStream_t s) {
                          traits::set_cold(field_.data(), field_.topology(), s);
                      }) {
            traits::set_cold(field_.data(), field_.topology(), nullptr);
            RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());
        } else {
            RETICOLO_CUDA_CHECK(
                cudaMemset(field_.data(), 0, field_.size() * sizeof(value_type)));
        }
    }
    void upload(Lattice<value_type> const& l) {
        field_.copy_from_host(l);
        RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());
    }

    // LLR hot-start: disorder the config before warm-in, mirroring the CPU
    // llr::Replica::hot_start. Opt-in — only actions whose device trait provides
    // hot_start (the gauge families) do anything; scalar actions cold-start fine
    // and this is a no-op for them. The RNG stream uses the replica's seed with a
    // dedicated counter, disjoint from the per-trajectory momentum stream.
    void hot_start(double sigma) {
        using traits = device_functors<HostAction>;
        if constexpr (requires(std::uint64_t const* c, cudaStream_t s) {
                          traits::hot_start(
                              field_.data(), static_cast<long>(field_.size()), sigma, seed_, c, s);
                      }) {
            DeviceBuffer<std::uint64_t> ctr{1};
            std::uint64_t const tag = k_hot_counter;
            RETICOLO_CUDA_CHECK(
                cudaMemcpy(ctr.data(), &tag, sizeof(tag), cudaMemcpyHostToDevice));
            traits::hot_start(
                field_.data(), static_cast<long>(field_.size()), sigma, seed_, ctr.data(), nullptr);
            RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // Async (no sync) so the driver can overlap replicas; ordered before the
    // measurement trajectories on the same stream, so no barrier is needed.
    void thermalize(int n) { hmc_.run(n); }

    // Host-free measurement block: begin → N × measure_trajectory → end. Every
    // trajectory's dE contribution accumulates on the device, so the whole block
    // (across all replicas) is enqueued before a single readback per replica.
    void begin_measure() { hmc_.begin_measure(); }
    void measure_trajectory() { hmc_.measure_trajectory(e_n_); }
    [[nodiscard]] double end_measure(int n_meas) { return hmc_.end_measure(n_meas); }

    // Exchange energy E = base action S of the current config (syncs).
    [[nodiscard]] double energy() { return hmc_.constraint_value(); }

    [[nodiscard]] double a() const noexcept { return hmc_.action().a(); }
    [[nodiscard]] double e_n() const noexcept { return e_n_; }
    [[nodiscard]] double delta() const noexcept { return delta_; }
    void set_a(double v) { hmc_.action().set_a(v); }

    // Param-swap exchange: swap {a, E_n} with a neighbour, re-upload both. Keeps
    // each config (and its baked graph pointer) in place; windows migrate across
    // slots (parallel-tempering style), so output is written per-slot.
    void swap_window(Replica& other) {
        double const a0 = a();
        double const e0 = e_n_;
        double const a1 = other.a();
        double const e1 = other.e_n_;
        hmc_.action().set_window(a1, e1);
        e_n_ = e1;
        other.hmc_.action().set_window(a0, e0);
        other.e_n_ = e0;
    }

    // Warm-in primitives, split so a driver can overlap replicas: launch a batch
    // of HMC (async, no sync), then check convergence (syncs this stream). Launch
    // every replica's batch before checking any and the batches overlap.
    void warm_launch(int batch) { hmc_.run(batch); }
    [[nodiscard]] bool warm_reached(double threshold_sigmas) {
        double const q     = hmc_.constraint_value();
        double const ratio = (delta_ != 0.0) ? std::abs(q - e_n_) / delta_ : 0.0;
        return ratio < threshold_sigmas;
    }

    [[nodiscard]] double acceptance() { return hmc_.acceptance(); }
    [[nodiscard]] Field& field() noexcept { return field_; }

private:
    // Philox counter for the hot-start draw — a fixed, large offset so it never
    // overlaps the per-trajectory momentum stream (which counts from 0).
    static constexpr std::uint64_t k_hot_counter = 1ULL << 48;

    Field field_;
    Hmc<Windowed, Integ, Field> hmc_;
    double e_n_;
    double delta_;
    std::uint64_t seed_;
};

}  // namespace reticolo::cuda::llr
