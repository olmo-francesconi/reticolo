#pragma once

// Device LLR windowed action — nvcc-only (.cuh; launches kernels). The GPU twin
// of llr::WindowedAction<Base>. Wraps a base DeviceAction and adds the
// Gaussian-penalty window; exposes exactly what cuda::Hmc consumes
// (sample_momenta / s_full_into / compute_force) plus constraint_s_full_into for
// the Hmc LLR readback.
//
// Mode A — real LLR (base action has no imaginary part). The window constrains
// the base action S:
//     S_LLR    = (1 + a) * S_base + (S_base - E_n)^2 / (2 * delta^2)
//     F_LLR(x) = force_scale(S_base) * F_base(x)
//
// Mode B — complex LLR (base DeviceAction satisfies HasImagDevice, i.e. exposes
// s_imag / compute_force_imag; BoseGas). HMC samples the real part S_R and the
// window constrains the imaginary observable q = S_I:
//     S_LLR    = S_R + a * S_I + (S_I - E_n)^2 / (2 * delta^2)
//     F_LLR(x) = F_R(x) + (a + (S_I - E_n)/delta^2) * F_I(x)
//     constraint = S_I   (the NR/RM loop watches <S_I - E_n>)
//
// Mode is chosen at compile time from the base's device interface (k_complex).
// The window parameters {a, E_n, delta} live in a device buffer (params_), read
// by the kernels — never baked as kernel literals, so set_a / set_window never
// forces a graph re-capture. Same discipline as the Philox trajectory counter.
// The value/scale math is the shared RETICOLO_HD formula
// (llr/formula/window_formula.hpp), one source of truth with the CPU.

#include <reticolo/core/field_traits.hpp>
#include <reticolo/cuda/check.hpp>
#include <reticolo/cuda/device_action.cuh>
#include <reticolo/cuda/device_buffer.hpp>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/device_topology.hpp>
#include <reticolo/cuda/reduce.cuh>
#include <reticolo/cuda/stream.hpp>
#include <reticolo/llr/formula/window_formula.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

#include <cuda_runtime.h>

namespace reticolo::cuda::llr {

// Mode A: out[0] = (1+a)·S + (S−E_n)²/2δ². params = {a, E_n, delta}. Single thread.
template <class Dummy = void>
__global__ void window_combine_kernel(double* out, double const* s_base, double const* params) {
    out[0] = reticolo::llr::formula::windowed_value(s_base[0], params[0], params[1], params[2]);
}

// Mode B: out[0] = S_R + a·S_I + (S_I−E_n)²/2δ². Single thread.
template <class Dummy = void>
__global__ void window_combine_complex_kernel(double* out,
                                              double const* s_r,
                                              double const* s_i,
                                              double const* params) {
    out[0] = reticolo::llr::formula::windowed_value_complex(
        s_r[0], s_i[0], params[0], params[1], params[2]);
}

// Mode A: force[i] *= force_scale(S_base). One thread per stored real element.
template <class T>
__global__ void scale_force_kernel(T* force, long n, double const* s_base, double const* params) {
    long const i = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (i >= n) {
        return;
    }
    double const scale =
        reticolo::llr::formula::force_scale(s_base[0], params[0], params[1], params[2]);
    force[i] = static_cast<T>(static_cast<double>(force[i]) * scale);
}

// Mode B: force[i] += scale·f_imag[i], scale = a + (S_I−E_n)/δ². One thread per
// underlying REAL element (the complex force reinterpreted as 2·nsites reals).
template <class T>
__global__ void merge_imag_force_kernel(
    T* force, T const* f_imag, long n, double const* s_i, double const* params) {
    long const i = (static_cast<long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (i >= n) {
        return;
    }
    double const scale =
        reticolo::llr::formula::force_scale_imag(s_i[0], params[0], params[1], params[2]);
    force[i] =
        static_cast<T>(static_cast<double>(force[i]) + (scale * static_cast<double>(f_imag[i])));
}

template <class HostAction, class Field>
class WindowedAction {
public:
    using value_type = typename Field::value_type;
    using real_t     = real_scalar_t<value_type>;

    static constexpr bool k_complex = HasImagDevice<HostAction, value_type>;
    // Flip to false to A/B the fused F_I+S_I path against the two-pass one
    // (measured ~16.5% faster wall-clock at L=12⁴ f64 on a T4).
    static constexpr bool k_fuse_imag = true;

    WindowedAction(HostAction host, DeviceTopology topo, double a, double e_n, double delta)
        : base_{host, topo}, params_{3}, s_tmp_{1}, s_i_{1},
          partials_{static_cast<std::size_t>(k_reduce_max_grid)}, a_h_{a}, e_n_h_{e_n},
          delta_h_{delta} {
        upload_();
        // Pre-allocate the mode-B F_I scratch NOW, not lazily inside compute_force
        // — that runs on the capture stream during MD, and a cudaMalloc there trips
        // "operation not permitted when stream is capturing" (the capture trap).
        if constexpr (k_complex) {
            f_imag_.emplace(topo);
        }
    }

    // --- interface cuda::Hmc consumes --------------------------------------
    void sample_momenta(value_type* mom,
                        long n,
                        std::uint64_t seed,
                        std::uint64_t const* traj,
                        cudaStream_t stream) const {
        base_.sample_momenta(mom, n, seed, traj, stream);
    }

    // Windowed S into out[0]. Mode A: reduce S_base, scalar combine. Mode B:
    // reduce S_R and S_I, complex combine.
    void s_full_into(double* out, Field const& field, double* partials, cudaStream_t stream) const {
        if constexpr (k_complex) {
            base_.s_full_into(s_tmp_.data(), field, partials, stream);  // S_R
            base_.s_imag_into(s_i_.data(), field, partials, stream);    // S_I
            window_combine_complex_kernel<void>
                <<<1, 1, 0, stream>>>(out, s_tmp_.data(), s_i_.data(), params_.data());
            RETICOLO_CUDA_CHECK_LAUNCH();
        } else {
            base_.s_full_into(s_tmp_.data(), field, partials, stream);
            window_combine_kernel<void><<<1, 1, 0, stream>>>(out, s_tmp_.data(), params_.data());
            RETICOLO_CUDA_CHECK_LAUNCH();
        }
    }

    // Windowed MD force, all on the current (capture) stream.
    void compute_force(Field const& field, Field& force) const {
        cudaStream_t const st = current_stream();
        if constexpr (k_complex) {
            base_.compute_force(field, force);  // F_R
            Field& fi = *f_imag_;               // pre-allocated in the constructor
            // Fused F_I + S_I in one τ-sweep when the base provides it; else the
            // two-pass (separate F_I gather + S_I reduction). k_fuse_imag flips
            // this for the A/B wall-clock measurement.
            if constexpr (k_fuse_imag && requires {
                              base_.compute_force_imag_and_s_imag_into(
                                  s_i_.data(), field, fi, partials_.data(), st);
                          }) {
                base_.compute_force_imag_and_s_imag_into(
                    s_i_.data(), field, fi, partials_.data(), st);
            } else {
                base_.compute_force_imag(field, fi);                          // F_I
                base_.s_imag_into(s_i_.data(), field, partials_.data(), st);  // S_I for the scale
            }
            auto const n_real    = static_cast<long>(force.size()) *
                                   static_cast<long>(sizeof(value_type) / sizeof(real_t));
            constexpr int kBlock = 256;
            auto const grid      = static_cast<unsigned>((n_real + kBlock - 1) / kBlock);
            merge_imag_force_kernel<real_t>
                <<<grid, kBlock, 0, st>>>(reinterpret_cast<real_t*>(force.data()),
                                          reinterpret_cast<real_t const*>(fi.data()),
                                          n_real,
                                          s_i_.data(),
                                          params_.data());
            RETICOLO_CUDA_CHECK_LAUNCH();
        } else {
            if constexpr (requires {
                              base_.s_full_and_force(
                                  s_tmp_.data(), field, force, partials_.data(), st);
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
    }

    // The LLR constraint value S read for the a-update. Mode A: the base action
    // S. Mode B: the imaginary observable S_I.
    void constraint_s_full_into(double* out,
                                Field const& field,
                                double* partials,
                                cudaStream_t stream) const {
        if constexpr (k_complex) {
            base_.s_imag_into(out, field, partials, stream);
        } else {
            base_.s_full_into(out, field, partials, stream);
        }
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
    void upload_() {
        std::array<double, 3> const h{a_h_, e_n_h_, delta_h_};
        RETICOLO_CUDA_CHECK(cudaMemcpy(
            params_.data(), h.data(), h.size() * sizeof(double), cudaMemcpyHostToDevice));
    }

    DeviceAction<HostAction, Field> base_;
    mutable DeviceBuffer<double> params_;    // {a, E_n, delta} device-resident
    mutable DeviceBuffer<double> s_tmp_;     // S_R / S_base scalar between reduce and combine/scale
    mutable DeviceBuffer<double> s_i_;       // S_I scalar (mode B)
    mutable DeviceBuffer<double> partials_;  // reduction scratch for compute_force's reductions
    mutable std::optional<Field> f_imag_;    // F_I scratch (mode B, allocated in ctor)
    double a_h_;
    double e_n_h_;
    double delta_h_;
};

}  // namespace reticolo::cuda::llr
