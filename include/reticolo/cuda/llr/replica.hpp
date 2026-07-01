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
#include <reticolo/cuda/check.hpp>
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
          e_n_{e_n}, delta_{delta} {
        cold_start();
    }

    Replica(Replica const&)            = delete;
    Replica& operator=(Replica const&) = delete;
    Replica(Replica&&)                 = delete;
    Replica& operator=(Replica&&)      = delete;
    ~Replica()                         = default;

    // Cold config (phi = 0). memset of doubles/floats to 0 is the 0.0 bit pattern.
    void cold_start() {
        RETICOLO_CUDA_CHECK(cudaMemset(field_.data(), 0, field_.size() * sizeof(value_type)));
    }
    void upload(Lattice<value_type> const& l) {
        field_.copy_from_host(l);
        RETICOLO_CUDA_CHECK(cudaDeviceSynchronize());
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

    // Batches of HMC until |S_base - E_n| < threshold_sigmas·delta or the budget
    // is spent. Returns batches consumed (== max_batches means not converged).
    int warm_into_window(int max_batches, int batch = 10, double threshold_sigmas = 1.0) {
        for (int b = 0; b < max_batches; ++b) {
            hmc_.run(batch);
            double const q     = hmc_.constraint_value();
            double const ratio = (delta_ != 0.0) ? std::abs(q - e_n_) / delta_ : 0.0;
            if (ratio < threshold_sigmas) {
                return b + 1;
            }
        }
        return max_batches;
    }

    [[nodiscard]] double acceptance() { return hmc_.acceptance(); }
    [[nodiscard]] Field& field() noexcept { return field_; }

private:
    Field field_;
    Hmc<Windowed, Integ, Field> hmc_;
    double e_n_;
    double delta_;
};

}  // namespace reticolo::cuda::llr
