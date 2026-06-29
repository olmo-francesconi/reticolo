#pragma once

namespace reticolo::cuda {

// Phase 1 (M1) compile gate. Returns true if the gauge action + drift code
// paths (s_full, compute_force, expi_lmul_slab — including the transcendental
// vec_libm fallbacks that replace Sleef under nvcc) COMPILE under nvcc and
// produce finite output on a tiny config. The point is that the defining TU
// (src/cuda/gauge_probe.cu) compiles at all; the runtime check is a bonus.
[[nodiscard]] bool gauge_headers_compile();

}  // namespace reticolo::cuda
