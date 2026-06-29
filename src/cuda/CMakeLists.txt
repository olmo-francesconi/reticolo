# reticolo::cuda — the only target compiled by nvcc and the only one that may
# include <cuda_runtime.h> / cuRAND / cub. CUDA headers must never leak into
# reticolo::core, mirroring how reticolo::io quarantines <hdf5.h>.
#
# It uses reticolo::core headers (header-only — nothing to link) via the include
# dir below. As of M1, <sleef.h> and the SIMD intrinsics are guarded out of
# vec_libm.hpp under __CUDACC__, so the gauge math headers parse under nvcc
# (proven by gauge_probe.cu). See docs/cuda_hmc_roadmap.md.

find_package(CUDAToolkit REQUIRED)

add_library(reticolo_cuda STATIC selftest.cu reduce.cu gauge_probe.cu stencil_probe.cu phi4_probe.cu)
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
