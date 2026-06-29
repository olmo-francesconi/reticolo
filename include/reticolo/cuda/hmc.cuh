#pragma once

// Generic device HMC — nvcc-only (.cuh; transitively launches kernels).
//
// Mirrors alg::Hmc::step() once. There is NO action or integrator switch: the
// integrator is the type parameter Integ (an unchanged alg::integ::* tag), and
// the action is any type exposing s_full(Field) / compute_force(Field, Field)
// — here cuda::DeviceAction. The MD loop reuses Leapfrog/Omelyan2/Omelyan4
// verbatim through the device drift_field/kick_add atoms (cuda/integ_ops.hpp,
// resolved by ADL). Momenta come from the device Philox sampler (Phase 2c); the
// MH accept is a one-scalar host decision over ΔH.
//
// Phase 2d: the MD trajectory is captured into a CUDA graph on the first step
// and replayed thereafter, amortizing the per-kernel launch overhead of the
// n_md × (force + kicks + drifts) sequence. Capture requires every MD kernel on
// one non-default stream, so the atoms / DeviceAction launch on the thread-local
// current stream (cuda/stream.hpp), which we point at md_stream_ for the
// captured region. Sampling and the H0/H1 reductions stay eager — the
// reductions sync internally, so capturing them needs a device-scalar reduction
// (a later perf step); the MD loop is the bulk of the launches.

#include <reticolo/algorithm/integrators.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_action.cuh>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/graph.hpp>
#include <reticolo/cuda/integ_ops.hpp>
#include <reticolo/cuda/reduce.hpp>
#include <reticolo/cuda/rng_philox.cuh>
#include <reticolo/cuda/stream.hpp>

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace reticolo::cuda {

struct HmcResult {
    double dH     = 0.0;
    bool accepted = false;
};

template <class A, class R, class Integ = alg::integ::Leapfrog, class Field = DeviceField<double>>
class Hmc {
public:
    Hmc(A action, Field& field, R& rng, double tau, int n_md, std::uint64_t seed = 0xC0FFEEULL)
        : action_{std::move(action)},
          field_{field},
          rng_{rng},
          mom_{field.topology()},
          force_{field.topology()},
          old_{field.topology()},
          tau_{tau},
          n_md_{n_md},
          seed_{seed},
          traj_buf_{1},
          partials_{static_cast<std::size_t>(k_reduce_max_grid)},
          eng_{4},
          md_stream_{make_stream_()},
          graph_{md_stream_} {}

    ~Hmc() { cudaStreamDestroy(md_stream_); }

    Hmc(Hmc const&)            = delete;
    Hmc& operator=(Hmc const&) = delete;
    Hmc(Hmc&&)                 = delete;
    Hmc& operator=(Hmc&&)      = delete;

    // One trajectory, one stream, ONE host sync. Sampling, both Hamiltonian
    // reductions, the MD graph and the rollback copy all enqueue on md_stream_;
    // the reductions write their scalars to a device buffer (no per-call malloc
    // or sync). The host reads the four energy scalars once at the end for the
    // MH accept — this is what removes the per-trajectory reduction floor.
    HmcResult step() {
        auto const n = static_cast<long>(mom_.size());
        ScopedStream const scope{md_stream_};

        sample_momenta_();
        reduce_sumsq_into(eng_.data() + 0, mom_.data(), n, partials_.data(), md_stream_);  // 2·kin0
        action_.s_full_into(eng_.data() + 1, field_, partials_.data(), md_stream_);        // pot0
        copy_device_(old_, field_);

        graph_.run([&] { Integ::run(action_, field_, mom_, force_, tau_, n_md_); });

        reduce_sumsq_into(eng_.data() + 2, mom_.data(), n, partials_.data(), md_stream_);  // 2·kin1
        action_.s_full_into(eng_.data() + 3, field_, partials_.data(), md_stream_);        // pot1

        eng_.copy_to_host(h_eng_.data(), md_stream_);
        RETICOLO_CUDA_CHECK(cudaStreamSynchronize(md_stream_));

        double const h0     = (0.5 * h_eng_[0]) + h_eng_[1];
        double const h1     = (0.5 * h_eng_[2]) + h_eng_[3];
        double const dH     = h1 - h0;
        bool const accepted = (dH <= 0.0) || (rng_.uniform() < std::exp(-dH));
        if (!accepted) {
            copy_device_(field_, old_);
            RETICOLO_CUDA_CHECK(cudaStreamSynchronize(md_stream_));
        }
        return {.dH = dH, .accepted = accepted};
    }

private:
    [[nodiscard]] static cudaStream_t make_stream_() {
        cudaStream_t s = nullptr;
        // Non-blocking: no implicit dependency on the legacy default stream,
        // which would otherwise trip stream capture. Ordering with the eager
        // (default-stream) sampling / reductions is enforced by explicit syncs.
        RETICOLO_CUDA_CHECK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
        return s;
    }

    // Fill momenta on md_stream_ — no sync; the single step() sync covers it.
    // The trajectory counter (device buffer) is bumped host-side and copied on
    // md_stream_ before the fill, ordered by the stream.
    void sample_momenta_() {
        traj_buf_.copy_from_host(&traj_, md_stream_);
        fill_normals(mom_.data(), static_cast<long>(mom_.size()), seed_, traj_buf_.data(),
                     md_stream_);
        ++traj_;
    }

    void copy_device_(Field& dst, Field const& src) {
        RETICOLO_CUDA_CHECK(cudaMemcpyAsync(dst.data(), src.data(),
                                            src.size() * sizeof(typename Field::value_type),
                                            cudaMemcpyDeviceToDevice, md_stream_));
    }

    A action_;
    Field& field_;
    R& rng_;
    Field mom_;
    Field force_;
    Field old_;
    double tau_;
    int n_md_;
    std::uint64_t seed_;
    std::uint64_t traj_ = 0;
    DeviceBuffer<std::uint64_t> traj_buf_;
    DeviceBuffer<double> partials_;        // reduction scratch (no per-step malloc)
    DeviceBuffer<double> eng_;             // device scalars: kin0, pot0, kin1, pot1
    std::array<double, 4> h_eng_{};        // their host mirror, copied once per step
    cudaStream_t md_stream_;
    TrajectoryGraph graph_;
};

}  // namespace reticolo::cuda
