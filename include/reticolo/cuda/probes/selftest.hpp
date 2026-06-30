#pragma once

namespace reticolo::cuda {

// Toolchain self-test. Allocates a device buffer, runs a trivial
// kernel on it, and round-trips through pinned host memory, asserting the data
// survives. Returns true on success. Defined in src/cuda/selftest.cu — the
// only thing the host side needs is this declaration, so a host TU (test/app)
// can call it without pulling in <cuda_runtime.h>.
[[nodiscard]] bool selftest();

}  // namespace reticolo::cuda
