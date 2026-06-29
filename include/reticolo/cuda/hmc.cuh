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
          md_stream_{make_stream_()},
          graph_{md_stream_} {}

    ~Hmc() { cudaStreamDestroy(md_stream_); }

    Hmc(Hmc const&)            = delete;
    Hmc& operator=(Hmc const&) = delete;
    Hmc(Hmc&&)                 = delete;
    Hmc& operator=(Hmc&&)      = delete;

    HmcResult step() {
        sample_momenta_();
        copy_device_(old_, field_);
        double const h0 = hamiltonian_();
        run_md_();
        double const h1     = hamiltonian_();
        double const dH     = h1 - h0;
        bool const accepted = (dH <= 0.0) || (rng_.uniform() < std::exp(-dH));
        if (!accepted) {
            copy_device_(field_, old_);
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

    // Capture the MD trajectory on md_stream_ once, replay it every step. The
    // baked device pointers (field_/mom_/force_) never move, so the graph stays
    // valid; n_md is fixed for this Hmc, so the node count never changes.
    void run_md_() {
        ScopedStream const scope{md_stream_};
        graph_.run([&] { Integ::run(action_, field_, mom_, force_, tau_, n_md_); });
        RETICOLO_CUDA_CHECK(cudaStreamSynchronize(md_stream_));
    }

    void sample_momenta_() {
        traj_buf_.copy_from_host(&traj_);
        fill_normals(mom_.data(), static_cast<long>(mom_.size()), seed_, traj_buf_.data());
        ++traj_;
        RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }

    [[nodiscard]] double hamiltonian_() {
        double const kinetic = 0.5 * reduce_sumsq_f64(mom_.data(), static_cast<long>(mom_.size()));
        double const potential = action_.s_full(field_);
        return kinetic + potential;
    }

    static void copy_device_(Field& dst, Field const& src) {
        RETICOLO_CUDA_CHECK(cudaMemcpyAsync(dst.data(), src.data(),
                                            src.size() * sizeof(typename Field::value_type),
                                            cudaMemcpyDeviceToDevice, nullptr));
        RETICOLO_CUDA_CHECK(cudaStreamSynchronize(nullptr));
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
    cudaStream_t md_stream_;
    TrajectoryGraph graph_;
};

}  // namespace reticolo::cuda
