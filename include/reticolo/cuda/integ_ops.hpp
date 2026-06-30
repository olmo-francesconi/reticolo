#pragma once

#include <reticolo/core/cplx.hpp>
#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/reduce.cuh>
#include <reticolo/cuda/stream.hpp>

#include <cuda_runtime.h>

// Device integrator atoms — the elementwise drift `field += c·mom` and the
// additive kick `mom += k·force`, for DeviceField. These are the device
// overloads of the CPU `alg::integ::drift_field` / `kick_add`
// (integ_ops.hpp); they live in `reticolo::cuda` so ADL finds them when the
// unchanged Leapfrog/Omelyan2/Omelyan4::run instantiate on a DeviceField, and
// partial ordering prefers them over the generic host loops (which would
// otherwise dereference device pointers on the host). They reduce to the
// already-deterministic `axpy_f64` — no integrator-specific kernel exists.
//
// f64 and f32 overloads. The MD step runs in field precision — the f32 atoms
// cast the integrator's double coefficient down to float — matching the CPU
// mixed-precision HMC; the Hamiltonian reductions still accumulate in double
// (reduce.cuh).

namespace reticolo::cuda {

template <class Layout>
inline void drift_field(DeviceField<double, Layout>& field,
                        DeviceField<double, Layout> const& mom,
                        double cdt) {
    axpy_f64(cdt, mom.data(), field.data(), static_cast<long>(field.size()), current_stream());
}

template <class Layout>
inline void
kick_add(DeviceField<double, Layout>& mom, DeviceField<double, Layout> const& force, double kdt) {
    axpy_f64(kdt, force.data(), mom.data(), static_cast<long>(mom.size()), current_stream());
}

template <class Layout>
inline void
drift_field(DeviceField<float, Layout>& field, DeviceField<float, Layout> const& mom, double cdt) {
    axpy_f32(static_cast<float>(cdt),
             mom.data(),
             field.data(),
             static_cast<long>(field.size()),
             current_stream());
}

template <class Layout>
inline void
kick_add(DeviceField<float, Layout>& mom, DeviceField<float, Layout> const& force, double kdt) {
    axpy_f32(static_cast<float>(kdt),
             force.data(),
             mom.data(),
             static_cast<long>(mom.size()),
             current_stream());
}

// Complex-field atoms (BoseGas): a cplx<T> field is 2 reals/site, so the additive
// MD drift/kick and the kinetic ½Σ|p|² are correct elementwise on the underlying
// reals. These reinterpret to the real buffer (2·size) and reuse the same
// deterministic primitives, so the generic Hmc / Integ::run work on a complex
// field unchanged. Declared here (which hmc.cuh includes) so reduce_sumsq_into's
// cplx overload is visible at the Hmc definition — cplx's only ADL-associated
// namespace is `reticolo`, not `reticolo::cuda`, so visibility, not ADL, finds it.
template <class T, class Layout>
inline void drift_field(DeviceField<cplx<T>, Layout>& field,
                        DeviceField<cplx<T>, Layout> const& mom,
                        double cdt) {
    auto* const f       = reinterpret_cast<T*>(field.data());
    auto const* const m = reinterpret_cast<T const*>(mom.data());
    long const n        = 2 * static_cast<long>(field.size());
    if constexpr (sizeof(T) == sizeof(double)) {
        axpy_f64(cdt, m, f, n, current_stream());
    } else {
        axpy_f32(static_cast<float>(cdt), m, f, n, current_stream());
    }
}
template <class T, class Layout>
inline void
kick_add(DeviceField<cplx<T>, Layout>& mom, DeviceField<cplx<T>, Layout> const& force, double kdt) {
    auto* const m       = reinterpret_cast<T*>(mom.data());
    auto const* const f = reinterpret_cast<T const*>(force.data());
    long const n        = 2 * static_cast<long>(mom.size());
    if constexpr (sizeof(T) == sizeof(double)) {
        axpy_f64(kdt, f, m, n, current_stream());
    } else {
        axpy_f32(static_cast<float>(kdt), f, m, n, current_stream());
    }
}

// Kinetic ½Σ|p|² = ½Σ(re²+im²): sum of squares over the 2·n underlying reals.
inline void reduce_sumsq_into(
    double* out, cplx<double> const* x, long n, double* partials, cudaStream_t stream = nullptr) {
    reduce_sumsq_into(out, reinterpret_cast<double const*>(x), 2 * n, partials, stream);
}
inline void reduce_sumsq_into(
    double* out, cplx<float> const* x, long n, double* partials, cudaStream_t stream = nullptr) {
    reduce_sumsq_into(out, reinterpret_cast<float const*>(x), 2 * n, partials, stream);
}

}  // namespace reticolo::cuda
