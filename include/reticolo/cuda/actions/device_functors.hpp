#pragma once

// The host-action → device-functor trait, declared once. Each device-ported
// action specializes device_functors<HostAction> (in its cuda/actions/<name>.hpp)
// to name its force/energy functor pair and build them from the host action's
// couplings. cuda::DeviceAction consumes the trait generically and never changes.

namespace reticolo::cuda {

template <class HostAction>
struct device_functors;

}  // namespace reticolo::cuda
