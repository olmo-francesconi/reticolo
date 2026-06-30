# reticolo::cuda — the only target compiled by nvcc and the only one that may
# include <cuda_runtime.h> / cuRAND / cub. CUDA headers must never leak into
# reticolo::core, mirroring how reticolo::io quarantines <hdf5.h>.
#
# It uses reticolo::core headers (header-only — nothing to link) via the include
# dir below. <sleef.h> and the SIMD intrinsics are guarded out of
# vec_libm.hpp under __CUDACC__, so the gauge math headers parse under nvcc
# (proven by gauge_probe.cu).

find_package(CUDAToolkit REQUIRED)

# reduce.cu is the only real-backend TU; everything under probes/ is the
# validation harness driven by tests/cuda/test_cuda_selftest.cpp.
add_library(reticolo_cuda STATIC
    reduce.cu
    probes/selftest.cu probes/gauge_probe.cu probes/stencil_probe.cu
    probes/phi4_probe.cu probes/scalar_probe.cu probes/f32_probe.cu
    probes/u1_probe.cu probes/wilson_u1_probe.cu probes/su2_probe.cu probes/su3_probe.cu
    probes/bose_gas_probe.cu probes/hmc_probe.cu probes/rng_probe.cu)
add_library(reticolo::cuda ALIAS reticolo_cuda)

target_include_directories(reticolo_cuda PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# CUDA runtime is propagated to host consumers that link the static archive.
target_link_libraries(reticolo_cuda PUBLIC CUDA::cudart)

set_target_properties(reticolo_cuda PROPERTIES
    CUDA_STANDARD          20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON
)
