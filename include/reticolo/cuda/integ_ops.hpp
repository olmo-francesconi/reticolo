#pragma once

#include <reticolo/cuda/device_field.hpp>
#include <reticolo/cuda/reduce.hpp>
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
// f64 (Phase 2) and f32 (Phase 3) overloads. The MD step runs in field
// precision — the f32 atoms cast the integrator's double coefficient down to
// float — matching the CPU mixed-precision HMC; the Hamiltonian reductions
// still accumulate in double (reduce.hpp).

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

}  // namespace reticolo::cuda
