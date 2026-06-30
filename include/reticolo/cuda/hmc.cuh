#pragma once

// Generic device HMC — nvcc-only (.cuh; transitively launches kernels).
//
// Mirrors alg::Hmc::step() once. There is NO action or integrator switch: the
// integrator is the type parameter Integ (an unchanged alg::integ::* tag), and
// the action is any type exposing s_full(Field) / compute_force(Field, Field)
// — here cuda::DeviceAction. The MD loop reuses Leapfrog/Omelyan2/Omelyan4
// verbatim through the device drift_field/kick_add atoms (cuda/integ_ops.hpp,
// resolved by ADL).
//
// HOST-FREE TRAJECTORIES. The ENTIRE trajectory — save, momentum sampling, both
// Hamiltonian reductions, the MD loop, the Metropolis accept and the conditional
// rollback — is captured into a single CUDA graph and replayed with one
// cudaGraphLaunch per trajectory. Nothing returns to the host between
// trajectories: the accept/reject decision is a device kernel (mh_accept_kernel)
// that draws its own Philox uniform on a separate key stream, the rollback is a
// per-site select kernel, and the Philox trajectory counter is bumped on the
// device. `run(k)` replays k trajectories without a single host sync; the host
// touches the chain only when it measures (`sync()` + read the field). This is
// what removes the per-trajectory host-sync floor.
//
// The capture trap (graph.hpp): anything that varies per trajectory must be read
// from a device buffer, never baked as a kernel literal. The trajectory counter
// is device-resident and bumped by bump_counter_kernel inside the captured body,
// so every replay reads a fresh counter and the chain is ergodic.

#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/core/philox.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_action.cuh>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/graph.hpp>
#include <reticolo/cuda/integ_ops.hpp>
#include <reticolo/cuda/reduce.cuh>
#include <reticolo/cuda/stream.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

#include <cuda_runtime.h>

namespace reticolo::cuda {

// All three kernels are templated purely for weak/mergeable linkage — a plain
// __global__ in a header included by multiple TUs collides at device link under
// -rdc=true (the same reason fill_normals_kernel<T> is a template).

// Metropolis accept on the device. eng = {2·kin0, pot0, 2·kin1, pot1}; H = ½·2kin
// + pot. The accept uniform comes from a SEPARATE Philox key stream (seed ^ salt)
// so it can never collide with the momentum draws keyed on `seed`. Single thread.
template <class Dummy = void>
__global__ void mh_accept_kernel(double const* eng,
                                 std::uint64_t const* traj,
                                 std::uint64_t seed,
                                 int* accept,
                                 std::uint64_t* acc_count) {
    double const h0 = (0.5 * eng[0]) + eng[1];
    double const h1 = (0.5 * eng[2]) + eng[3];
    double const dH = h1 - h0;
    double u0       = 0.0;
    double u1       = 0.0;
    philox_uniform2(seed ^ 0x9E3779B97F4A7C15ULL, *traj, 0ULL, u0, u1);
    int const a = ((dH <= 0.0) || (u0 < exp(-dH))) ? 1 : 0;
    *accept     = a;
    *acc_count += static_cast<std::uint64_t>(a);
}

// Conditional rollback: on reject (accept==0) restore the saved config; on accept
// leave the proposal in place (the saved buffer is overwritten by the next
// trajectory's save). One thread per site, branch on the device-resident flag.
template <class T>
__global__ void resolve_kernel(T* field, T const* saved, int const* accept, long n) {
    long const i = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (i >= n) {
        return;
    }
    if (*accept == 0) {
        field[i] = saved[i];
    }
}

template <class Dummy = void>
__global__ void bump_counter_kernel(std::uint64_t* traj) {
    ++(*traj);
}

struct HmcResult {
    double dH     = 0.0;
    bool accepted = false;
};

template <class A, class Integ = alg::integ::Leapfrog, class Field = DeviceField<double>>
class Hmc {
public:
    Hmc(A action, Field& field, double tau, int n_md, std::uint64_t seed = 0xC0FFEEULL)
        : action_{std::move(action)}, field_{field}, mom_{field.topology()},
          force_{field.topology()}, old_{field.topology()}, tau_{tau}, n_md_{n_md}, seed_{seed},
          traj_buf_{1}, acc_buf_{1}, accept_buf_{1},
          partials_{static_cast<std::size_t>(k_reduce_max_grid)}, eng_{4},
          md_stream_{make_stream_()}, graph_{md_stream_} {
        std::uint64_t const zero = 0;
        traj_buf_.copy_from_host(&zero, md_stream_);
        acc_buf_.copy_from_host(&zero, md_stream_);
        RETICOLO_CUDA_CHECK(cudaStreamSynchronize(md_stream_));
    }

    ~Hmc() { cudaStreamDestroy(md_stream_); }

    Hmc(Hmc const&)            = delete;
    Hmc& operator=(Hmc const&) = delete;
    Hmc(Hmc&&)                 = delete;
    Hmc& operator=(Hmc&&)      = delete;

    // Replay k full trajectories host-free: each is one cudaGraphLaunch, no host
    // sync, no readback. The accept count accumulates on the device. Sync only
    // when you want to measure.
    void run(int k) {
        ScopedStream const scope{md_stream_};
        for (int i = 0; i < k; ++i) {
            graph_.run([&] { enqueue_trajectory_(); });
        }
        traj_count_ += static_cast<std::uint64_t>(k);
    }

    void sync() { RETICOLO_CUDA_CHECK(cudaStreamSynchronize(md_stream_)); }

    // One trajectory with host-visible diagnostics (the per-step probe path):
    // replay once, copy the energies + accept flag back, sync once.
    HmcResult step() {
        run(1);
        eng_.copy_to_host(h_eng_.data(), md_stream_);
        accept_buf_.copy_to_host(&h_accept_, md_stream_);
        sync();
        double const h0 = (0.5 * h_eng_[0]) + h_eng_[1];
        double const h1 = (0.5 * h_eng_[2]) + h_eng_[3];
        return {.dH = h1 - h0, .accepted = h_accept_ != 0};
    }

    // Cumulative acceptance over every trajectory run so far. Syncs.
    [[nodiscard]] double acceptance() {
        std::uint64_t accepted = 0;
        acc_buf_.copy_to_host(&accepted, md_stream_);
        sync();
        return traj_count_ == 0 ? 0.0
                                : static_cast<double>(accepted) / static_cast<double>(traj_count_);
    }

    // --- checkpoint hooks ---------------------------------------------------
    // The complete per-run RNG state is (seed_, device trajectory counter): the
    // Philox stream is stateless given (seed, counter, pair). Saving these two +
    // the field lets a resumed run reproduce an uninterrupted one bit-for-bit.
    [[nodiscard]] std::uint64_t seed() const { return seed_; }

    // The authoritative Philox counter (device-resident, bumped per trajectory).
    // Reads back from the device — syncs.
    [[nodiscard]] std::uint64_t rng_counter() {
        std::uint64_t c = 0;
        traj_buf_.copy_to_host(&c, md_stream_);
        sync();
        return c;
    }

    // Restore the Philox counter on resume so the next trajectory draws the same
    // momenta / MH uniforms as the uninterrupted run would have. The acceptance
    // accumulator (acc_buf_ / traj_count_) intentionally restarts this session.
    void set_rng_counter(std::uint64_t c) {
        traj_buf_.copy_from_host(&c, md_stream_);
        sync();
    }

private:
    [[nodiscard]] static cudaStream_t make_stream_() {
        cudaStream_t s = nullptr;
        // Non-blocking: no implicit dependency on the legacy default stream,
        // which would otherwise trip stream capture.
        RETICOLO_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        return s;
    }

    // The full per-trajectory operation sequence, all on md_stream_. Captured
    // once by graph_.run and replayed thereafter. Every per-trajectory-varying
    // input (the Philox counter) is read from a device buffer, never baked.
    void enqueue_trajectory_() {
        auto const n         = static_cast<long>(mom_.size());
        constexpr int kBlock = 256;
        auto const site_grid = static_cast<unsigned>((n + kBlock - 1) / kBlock);

        copy_device_(old_, field_);  // save q0 for a possible rollback
        action_.sample_momenta(mom_.data(), n, seed_, traj_buf_.data(), md_stream_);

        reduce_sumsq_into(eng_.data() + 0, mom_.data(), n, partials_.data(), md_stream_);  // 2·kin0
        action_.s_full_into(eng_.data() + 1, field_, partials_.data(), md_stream_);        // pot0

        Integ::run(action_, field_, mom_, force_, tau_, n_md_);

        reduce_sumsq_into(eng_.data() + 2, mom_.data(), n, partials_.data(), md_stream_);  // 2·kin1
        action_.s_full_into(eng_.data() + 3, field_, partials_.data(), md_stream_);        // pot1

        mh_accept_kernel<void><<<1, 1, 0, md_stream_>>>(
            eng_.data(), traj_buf_.data(), seed_, accept_buf_.data(), acc_buf_.data());
        RETICOLO_CUDA_CHECK_LAUNCH();
        resolve_kernel<typename Field::value_type><<<site_grid, kBlock, 0, md_stream_>>>(
            field_.data(), old_.data(), accept_buf_.data(), n);
        RETICOLO_CUDA_CHECK_LAUNCH();
        bump_counter_kernel<void><<<1, 1, 0, md_stream_>>>(traj_buf_.data());
        RETICOLO_CUDA_CHECK_LAUNCH();
    }

    void copy_device_(Field& dst, Field const& src) {
        RETICOLO_CUDA_CHECK(cudaMemcpyAsync(dst.data(),
                                            src.data(),
                                            src.size() * sizeof(typename Field::value_type),
                                            cudaMemcpyDeviceToDevice,
                                            md_stream_));
    }

    A action_;
    Field& field_;
    Field mom_;
    Field force_;
    Field old_;
    double tau_;
    int n_md_;
    std::uint64_t seed_;
    std::uint64_t traj_count_ = 0;
    DeviceBuffer<std::uint64_t> traj_buf_;  // device trajectory counter (bumped on device)
    DeviceBuffer<std::uint64_t> acc_buf_;   // device accepted-count accumulator
    DeviceBuffer<int> accept_buf_;          // per-trajectory accept flag (device)
    DeviceBuffer<double> partials_;         // reduction scratch (no per-step malloc)
    DeviceBuffer<double> eng_;              // device scalars: 2·kin0, pot0, 2·kin1, pot1
    std::array<double, 4> h_eng_{};         // host mirror for step()
    int h_accept_ = 0;                      // host mirror of accept_buf_ for step()
    cudaStream_t md_stream_;
    TrajectoryGraph graph_;
};

}  // namespace reticolo::cuda
