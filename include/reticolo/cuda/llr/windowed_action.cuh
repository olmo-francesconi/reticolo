#pragma once

// Device LLR windowed action — nvcc-only (.cuh; launches kernels). The GPU twin
// of llr::WindowedAction<Base> (mode A only). Wraps a base DeviceAction and adds
// the Gaussian-penalty window; exposes exactly what cuda::Hmc consumes
// (sample_momenta / s_full_into / compute_force) plus constraint_s_full_into for
// the Hmc LLR readback.
//
//     S_LLR    = (1 + a) * S_base + (S_base - E_n)^2 / (2 * delta^2)
//     F_LLR(x) = force_scale(S_base) * F_base(x),
//                force_scale = (1 + a) + (S_base - E_n) / delta^2
//
// The window scale is GLOBAL — it depends on the total action S_base, a
// reduction — so compute_force is: base force -> a fresh base-S reduction into a
// device scalar -> a broadcast scale kernel. All three enqueue on the capture
// stream with no host sync, so the whole thing stays inside cuda::Hmc's graph.
//
// The window parameters {a, E_n, delta} live in a device buffer (params_), read
// by the kernels — never baked as kernel literals, so set_a / set_window (an
// H2D of a few doubles) never forces a graph re-capture. Same discipline as the
// Philox trajectory counter. The value/scale math is the shared RETICOLO_HD
// formula (llr/detail/window_formula.hpp), one source of truth with the CPU.

#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_action.cuh>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/reduce.cuh>
#include <reticolo/cuda/stream.hpp>
#include <reticolo/llr/detail/window_formula.hpp>

#include <array>
#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace reticolo::cuda::llr {

// out[0] = (1+a)·S + (S−E_n)²/2δ².  s_base[0] is the reduced base action;
// params = {a, E_n, delta}. Single thread (a scalar combine). Templated purely
// for weak/mergeable linkage under -rdc=true (as the hmc.cuh kernels are).
template <class Dummy = void>
__global__ void window_combine_kernel(double* out, double const* s_base, double const* params) {
    double const s = s_base[0];
    out[0]         = reticolo::llr::detail::windowed_value(s, params[0], params[1], params[2]);
}

// force[i] *= force_scale(S_base). One thread per stored element (real scaling
// of the field's raw storage — scalar or, later, per-real-component gauge).
template <class T>
__global__ void scale_force_kernel(T* force, long n, double const* s_base, double const* params) {
    long const i = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (i >= n) {
        return;
    }
    double const scale =
        reticolo::llr::detail::force_scale(s_base[0], params[0], params[1], params[2]);
    force[i] = static_cast<T>(static_cast<double>(force[i]) * scale);
}

template <class HostAction, class Field>
class WindowedAction {
public:
    using value_type = typename Field::value_type;

    WindowedAction(HostAction host, DeviceTopology topo, double a, double e_n, double delta)
        : base_{host, topo}, params_{3}, s_tmp_{1},
          partials_{static_cast<std::size_t>(k_reduce_max_grid)}, a_h_{a}, e_n_h_{e_n},
          delta_h_{delta} {
        upload_();
    }

    // --- interface cuda::Hmc consumes --------------------------------------
    void sample_momenta(value_type* mom,
                        long n,
                        std::uint64_t seed,
                        std::uint64_t const* traj,
                        cudaStream_t stream) const {
        base_.sample_momenta(mom, n, seed, traj, stream);
    }

    // Windowed S into out[0]: reduce base S into s_tmp_, then the scalar combine.
    void s_full_into(double* out, Field const& field, double* partials, cudaStream_t stream) const {
        base_.s_full_into(s_tmp_.data(), field, partials, stream);
        window_combine_kernel<void><<<1, 1, 0, stream>>>(out, s_tmp_.data(), params_.data());
        RETICOLO_CUDA_CHECK_LAUNCH();
    }

    // F_LLR = force_scale(S_base) · F_base, all on the current (capture) stream.
    // S_base and F_base come from ONE field gather when the base action provides
    // the fused launcher (site actions), else a separate force + base-S reduction.
    void compute_force(Field const& field, Field& force) const {
        cudaStream_t const st = current_stream();
        if constexpr (requires {
                          base_.s_full_and_force(s_tmp_.data(), field, force, partials_.data(), st);
                      }) {
            base_.s_full_and_force(s_tmp_.data(), field, force, partials_.data(), st);
        } else {
            base_.compute_force(field, force);
            base_.s_full_into(s_tmp_.data(), field, partials_.data(), st);
        }
        auto const n         = static_cast<long>(force.size());
        constexpr int kBlock = 256;
        auto const grid      = static_cast<unsigned>((n + kBlock - 1) / kBlock);
        scale_force_kernel<value_type>
            <<<grid, kBlock, 0, st>>>(force.data(), n, s_tmp_.data(), params_.data());
        RETICOLO_CUDA_CHECK_LAUNCH();
    }

    // Mode A: the LLR constraint IS the base action S (no window).
    void constraint_s_full_into(double* out,
                                Field const& field,
                                double* partials,
                                cudaStream_t stream) const {
        base_.s_full_into(out, field, partials, stream);
    }

    // --- host window controls (graph-safe: params_ pointer is stable) ------
    void set_a(double a) {
        a_h_ = a;
        upload_();
    }
    void set_window(double a, double e_n) {
        a_h_   = a;
        e_n_h_ = e_n;
        upload_();
    }
    [[nodiscard]] double a() const noexcept { return a_h_; }
    [[nodiscard]] double e_n() const noexcept { return e_n_h_; }
    [[nodiscard]] double delta() const noexcept { return delta_h_; }

private:
    // Blocking H2D so the update is ordered before any subsequent stream work
    // (set_a / set_window run on the host between graph launches, never during
    // capture — a blocking copy here is fine and needs no stream bookkeeping).
    void upload_() {
        std::array<double, 3> const h{a_h_, e_n_h_, delta_h_};
        RETICOLO_CUDA_CHECK(cudaMemcpy(
            params_.data(), h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice));
    }

    DeviceAction<HostAction, Field> base_;
    mutable DeviceBuffer<double> params_;    // {a, E_n, delta} device-resident
    mutable DeviceBuffer<double> s_tmp_;     // base-S scalar between reduce and combine/scale
    mutable DeviceBuffer<double> partials_;  // reduction scratch for compute_force's base-S
    double a_h_;
    double e_n_h_;
    double delta_h_;
};

}  // namespace reticolo::cuda::llr
